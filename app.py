"""
Serveur Python - Système de Gestion de Parking Intelligent
Couche Serveur : Flask + ML (scikit-learn) + LLM (Claude API) + MongoDB
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import threading
import time
import logging
import os

from database import db_manager
from ml_model import ParkingPredictor
from llm_service import LLMService

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://ajmisouhail.github.io",
])
# ── Services ─────────────────────────────────────────────────────────────────
# ── Initialisation globale (pour gunicorn) ───────────────────────────────────
predictor = ParkingPredictor()
predictor.train_initial()          # ← ajouter
llm = LLMService()

# État courant (in-memory cache)
current_state = {
    "spots": {
        "P1": {"occupied": False, "distance": 200.0, "last_update": None},
        "P2": {"occupied": False, "distance": 200.0, "last_update": None},
        "P3": {"occupied": False, "distance": 200.0, "last_update": None},
        "P4": {"occupied": False, "distance": 200.0, "last_update": None},
    },
    "prediction": None,
    "last_sensor_update": None,
}
state_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Routes IoT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/sensors", methods=["POST"])
def receive_sensor_data():
    """Reçoit les données de l'ESP32 (capteurs HC-SR04)."""
    try:
        data = request.get_json(force=True)
        if not data or "spots" not in data:
            return jsonify({"error": "Payload invalide"}), 400

        ts = datetime.utcnow()
        with state_lock:
            for spot in data["spots"]:
                sid = spot["id"]
                if sid in current_state["spots"]:
                    current_state["spots"][sid].update({
                        "occupied":    spot["occupied"],
                        "distance":    spot["distance"],
                        "last_update": ts.isoformat(),
                    })
            current_state["last_sensor_update"] = ts.isoformat()

        # Persistance MongoDB
        record = {
            "timestamp":   ts,
            "device_id":   data.get("device_id", "unknown"),
            "spots":       data["spots"],
            "occupancy_rate": sum(1 for s in data["spots"] if s["occupied"]) / len(data["spots"]),
        }
        db_manager.insert_sensor_record(record)

        # Mise à jour prédiction
        _refresh_prediction()

        occupied = sum(1 for s in data["spots"] if s["occupied"])
        logger.info(f"[IoT] {occupied}/{len(data['spots'])} places occupées")
        return jsonify({"status": "ok", "received": len(data["spots"])}), 200

    except Exception as e:
        logger.error(f"[IoT] Erreur: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sensors/simulate", methods=["POST"])
def simulate_sensor():
    """Simule une mise à jour capteur (pour les tests sans ESP32)."""
    data = request.get_json(force=True)
    spot_id   = data.get("spot_id")
    occupied  = data.get("occupied", False)
    distance  = 5.0 if occupied else 200.0

    with state_lock:
        if spot_id in current_state["spots"]:
            current_state["spots"][spot_id].update({
                "occupied":    occupied,
                "distance":    distance,
                "last_update": datetime.utcnow().isoformat(),
            })

    _refresh_prediction()
    return jsonify({"status": "ok", "spot": spot_id, "occupied": occupied})


# ─────────────────────────────────────────────────────────────────────────────
# Routes état parking
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/parking/state", methods=["GET"])
def get_parking_state():
    """Retourne l'état complet du parking."""
    # Forcer une mise à jour simulée à chaque requête
    import random
    from datetime import datetime as dt
    hour = dt.now().hour
    if 8 <= hour <= 10 or 12 <= hour <= 14 or 17 <= hour <= 19:
        base_prob = 0.75
    elif 0 <= hour <= 6:
        base_prob = 0.1
    else:
        base_prob = 0.3

    with state_lock:
        for sid in ["P1", "P2", "P3", "P4"]:
            occupied = random.random() < base_prob
            current_state["spots"][sid].update({
                "occupied":    occupied,
                "distance":    round(random.uniform(3, 10), 1) if occupied else round(random.uniform(150, 250), 1),
                "last_update": dt.utcnow().isoformat(),
            })
        current_state["last_sensor_update"] = dt.utcnow().isoformat()

    _refresh_prediction()

    with state_lock:
        spots      = current_state["spots"]
        prediction = current_state["prediction"]

    total    = len(spots)
    occupied = sum(1 for s in spots.values() if s["occupied"])
    free     = total - occupied
    occ_rate = round(occupied / total * 100, 1) if total else 0

    return jsonify({
        "spots":              spots,
        "summary": {
            "total":          total,
            "occupied":       occupied,
            "free":           free,
            "occupancy_rate": occ_rate,
        },
        "prediction":         prediction,
        "last_update":        current_state["last_sensor_update"],
        "timestamp":          dt.utcnow().isoformat(),
    })


@app.route("/api/parking/history", methods=["GET"])
def get_parking_history():
    """Retourne l'historique d'occupation (séries temporelles)."""
    hours = int(request.args.get("hours", 24))
    since = datetime.utcnow() - timedelta(hours=hours)
    records = db_manager.get_records_since(since)

    series = [{
        "timestamp":      r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else r["timestamp"],
        "occupancy_rate": r.get("occupancy_rate", 0),
        "spots":          r.get("spots", []),
    } for r in records]

    return jsonify({"history": series, "count": len(series)})


@app.route("/api/parking/stats", methods=["GET"])
def get_stats():
    """Statistiques agrégées."""
    stats = db_manager.get_daily_stats()
    return jsonify(stats)


# ─────────────────────────────────────────────────────────────────────────────
# Routes LLM / IA
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    """Interface de discussion avec le LLM pour interroger les prévisions."""
    try:
        data     = request.get_json(force=True)
        question = data.get("message", "").strip()
        history  = data.get("history", [])

        if not question:
            return jsonify({"error": "Message vide"}), 400

        with state_lock:
            state = {
                "spots":      current_state["spots"],
                "prediction": current_state["prediction"],
            }

        response = llm.chat(question, state, history)
        return jsonify({"response": response, "timestamp": datetime.utcnow().isoformat()})

    except Exception as e:
        logger.error(f"[LLM] Erreur: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/llm/guidance", methods=["GET"])
def get_guidance():
    """Génère des consignes de guidage textuelles adaptées au trafic."""
    with state_lock:
        state = {
            "spots":      current_state["spots"],
            "prediction": current_state["prediction"],
        }
    guidance = llm.generate_guidance(state)
    return jsonify({"guidance": guidance, "timestamp": datetime.utcnow().isoformat()})


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_prediction():
    """Recalcule la prédiction ML et l'insère dans l'état courant."""
    with state_lock:
        spots = current_state["spots"]
        occupied = sum(1 for s in spots.values() if s["occupied"])
        occ_rate = occupied / len(spots)

    prediction = predictor.predict(occ_rate)

    with state_lock:
        current_state["prediction"] = prediction


def _background_simulator():
    """Simule des données IoT quand l'ESP32 n'est pas connecté."""
    logger.info("[SIM] Démarrage du simulateur IoT de secours")
    import random
    spot_ids = ["P1", "P2", "P3", "P4"]

    while True:
        time.sleep(5)
        # Simulation réaliste : plus d'occupation aux heures de pointe
        hour = datetime.now().hour
        base_prob = 0.3
        if 8 <= hour <= 10 or 12 <= hour <= 14 or 17 <= hour <= 19:
            base_prob = 0.75
        elif 0 <= hour <= 6:
            base_prob = 0.1

        for sid in spot_ids:
            occupied = random.random() < base_prob
            with state_lock:
                current_state["spots"][sid].update({
                    "occupied":    occupied,
                    "distance":    random.uniform(3, 10) if occupied else random.uniform(150, 250),
                    "last_update": datetime.utcnow().isoformat(),
                })
            current_state["last_sensor_update"] = datetime.utcnow().isoformat()

        _refresh_prediction()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "services": {
            "database": db_manager.is_connected(),
            "ml_model": predictor.is_trained(),
            "llm":      llm.is_available(),
        },
        "timestamp": datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor.train_initial()   # ← ajouter cette ligne
    _refresh_prediction()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, load_dotenv=False)

    # Entraîner le modèle ML avec les données historiques
    predictor.train_initial()

    # Prédiction initiale
    _refresh_prediction()

    # Simulateur de secours (si pas d'ESP32)
    if os.getenv("ENABLE_SIMULATOR", "true").lower() == "true":
        sim_thread = threading.Thread(target=_background_simulator, daemon=True)
        sim_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, load_dotenv=False)
