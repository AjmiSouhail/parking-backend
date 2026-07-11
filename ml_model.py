"""
Modèle ML de prédiction de saturation du parking
Utilise un RandomForestRegressor + features temporelles
"""

import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn non disponible, modèle simplifié utilisé")


class ParkingPredictor:
    """Prédit le taux d'occupation futur du parking (15 min, 30 min, 1h)."""

    def __init__(self):
        self._trained = False
        self._model   = None
        self._scaler  = None

    # ── Entraînement ─────────────────────────────────────────────────────────

    def train_initial(self):
        """Génère des données synthétiques réalistes et entraîne le modèle."""
        if not SKLEARN_AVAILABLE:
            self._trained = True   # fallback heuristique
            return

        logger.info("[ML] Génération des données d'entraînement synthétiques...")
        X, y = self._generate_training_data()

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        self._model.fit(X, y)
        self._trained = True
        logger.info(f"[ML] Modèle entraîné sur {len(X)} échantillons")

    def _generate_training_data(self, n_samples: int = 5000):
        """Crée un dataset synthétique avec patterns horaires réalistes."""
        X, y = [], []
        for _ in range(n_samples):
            hour        = np.random.randint(0, 24)
            minute      = np.random.randint(0, 60)
            weekday     = np.random.randint(0, 7)
            current_occ = np.random.uniform(0, 1)

            # Patterns : rush hours
            if hour in (8, 9, 12, 13, 17, 18):
                target = np.clip(current_occ + np.random.uniform(0.1, 0.3), 0, 1)
            elif hour in (0, 1, 2, 3, 4, 5):
                target = np.clip(current_occ - np.random.uniform(0.1, 0.2), 0, 1)
            elif weekday >= 5:   # weekend
                target = np.clip(current_occ + np.random.uniform(-0.05, 0.1), 0, 1)
            else:
                target = np.clip(current_occ + np.random.uniform(-0.05, 0.05), 0, 1)

            X.append([hour, minute, weekday, current_occ,
                       np.sin(2 * np.pi * hour / 24),
                       np.cos(2 * np.pi * hour / 24),
                       float(weekday < 5)])
            y.append(target)

        return np.array(X), np.array(y)

    # ── Prédiction ───────────────────────────────────────────────────────────

    def predict(self, current_occupancy_rate: float) -> dict:
        """Retourne les prédictions pour +15 min, +30 min, +60 min."""
        now     = datetime.now()
        hour    = now.hour
        minute  = now.minute
        weekday = now.weekday()

        horizons = {"15min": 0.25, "30min": 0.5, "60min": 1.0}
        results  = {}

        for label, delta_h in horizons.items():
            if not self._trained:
                results[label] = current_occupancy_rate
                continue

            if SKLEARN_AVAILABLE and self._model is not None:
                future_hour = (hour + delta_h) % 24
                feat = np.array([[
                    future_hour, minute, weekday,
                    current_occupancy_rate,
                    np.sin(2 * np.pi * future_hour / 24),
                    np.cos(2 * np.pi * future_hour / 24),
                    float(weekday < 5),
                ]])
                pred = float(np.clip(self._model.predict(feat)[0], 0, 1))
            else:
                pred = self._heuristic_predict(current_occupancy_rate, hour, delta_h, weekday)

            results[label] = round(pred * 100, 1)   # en %

        # Statut de saturation
        max_pred = max(results.values())
        if max_pred >= 90:
            status = "CRITIQUE"
            color  = "#ef4444"
        elif max_pred >= 70:
            status = "ÉLEVÉ"
            color  = "#f97316"
        elif max_pred >= 50:
            status = "MODÉRÉ"
            color  = "#eab308"
        else:
            status = "FAIBLE"
            color  = "#22c55e"

        return {
            "current":        round(current_occupancy_rate * 100, 1),
            "predictions":    results,
            "saturation_status": status,
            "saturation_color":  color,
            "computed_at":    now.isoformat(),
        }

    def _heuristic_predict(self, occ, hour, delta_h, weekday):
        future_h = (hour + delta_h) % 24
        rush = future_h in (8, 9, 12, 13, 17, 18)
        night = future_h in range(0, 6)
        if rush:
            return min(occ + 0.2, 1.0)
        if night:
            return max(occ - 0.15, 0.0)
        return occ

    def is_trained(self) -> bool:
        return self._trained
