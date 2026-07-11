"""
Service LLM - Génération de consignes et chat via Groq API (gratuit)
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("[LLM] groq non installé — pip install groq")


SYSTEM_PROMPT = """Tu es PARKIA, un assistant IA intelligent spécialisé dans la gestion d'un parking urbain.
Tu as accès en temps réel à :
- L'état de chaque place de parking (P1 à P4) : occupée ou libre
- Le taux d'occupation actuel et les prédictions ML pour +15min, +30min, +1h
- L'historique de fréquentation

Ton rôle :
1. Répondre aux questions des usagers sur la disponibilité et les prévisions de trafic
2. Donner des conseils de stationnement personnalisés et intelligents
3. Alerter sur les risques de saturation imminente
4. Proposer des itinéraires alternatifs si le parking est saturé
5. Formuler des consignes claires et adaptées au trafic en temps réel

Réponds toujours en français, de façon concise et utile. Utilise des émojis pour rendre les messages plus lisibles."""


class LLMService:

    def __init__(self):
        self._client = None
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._client = Groq(api_key=api_key)
                logger.info("[LLM] Client Groq initialisé")
            else:
                logger.warning("[LLM] GROQ_API_KEY non définie, mode heuristique")
        else:
            logger.warning("[LLM] groq non installé, mode heuristique")

    # ── Chat ─────────────────────────────────────────────────────────────────

    def chat(self, question: str, parking_state: dict, history: list) -> str:
        context = self._build_context(parking_state)
        full_question = f"{context}\n\nQuestion de l'usager : {question}"

        if self._client:
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in history[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": full_question})

                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=500,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"[LLM] Erreur API: {e}")

        return self._heuristic_response(question, parking_state)

    # ── Guidage ──────────────────────────────────────────────────────────────

    def generate_guidance(self, parking_state: dict) -> str:
        context = self._build_context(parking_state)
        prompt = f"{context}\n\nGénère une consigne de guidage courte (2-3 phrases) pour les usagers qui arrivent maintenant."

        if self._client:
            try:
                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"[LLM] Erreur guidage: {e}")

        return self._heuristic_guidance(parking_state)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_context(self, state: dict) -> str:
        spots = state.get("spots", {})
        pred  = state.get("prediction", {})

        lines = ["=== ÉTAT ACTUEL DU PARKING ==="]
        for sid, s in spots.items():
            status = "🔴 OCCUPÉE" if s["occupied"] else "🟢 LIBRE"
            lines.append(f"  Place {sid}: {status} (distance: {s.get('distance', '?')} cm)")

        if pred:
            lines.append(f"\n📊 Taux d'occupation: {pred.get('current', '?')}%")
            preds = pred.get("predictions", {})
            if preds:
                lines.append(f"  Prédictions ML:")
                lines.append(f"    +15 min: {preds.get('15min', '?')}%")
                lines.append(f"    +30 min: {preds.get('30min', '?')}%")
                lines.append(f"    +60 min: {preds.get('60min', '?')}%")
            lines.append(f"  Statut: {pred.get('saturation_status', '?')}")

        lines.append(f"  Horodatage: {datetime.now().strftime('%H:%M le %d/%m/%Y')}")
        return "\n".join(lines)

    def _heuristic_response(self, question: str, state: dict) -> str:
        spots = state.get("spots", {})
        free  = [k for k, v in spots.items() if not v["occupied"]]
        pred  = state.get("prediction", {})
        occ   = pred.get("current", 0)

        q = question.lower()
        if any(w in q for w in ["libre", "disponible", "place"]):
            if free:
                return f"🟢 Il y a {len(free)} place(s) libre(s) : {', '.join(free)}. Taux d'occupation : {occ}%."
            return f"🔴 Toutes les places sont occupées ({occ}%). Revenez dans 15 minutes."
        if any(w in q for w in ["prédiction", "prévision", "futur", "quand"]):
            preds = pred.get("predictions", {})
            return (f"📈 Prédictions : +15min → {preds.get('15min','?')}%, "
                    f"+30min → {preds.get('30min','?')}%, "
                    f"+1h → {preds.get('60min','?')}%.")
        return f"🅿️ Parking : {len(free)}/4 places libres, occupation à {occ}%."

    def _heuristic_guidance(self, state: dict) -> str:
        spots = state.get("spots", {})
        free  = [k for k, v in spots.items() if not v["occupied"]]
        pred  = state.get("prediction", {})
        occ   = pred.get("current", 0)

        if not free:
            return "🔴 Parking complet. Veuillez patienter ou chercher un stationnement alternatif."
        if occ > 70:
            return f"🟡 Parking presque complet ({occ}%). Places {', '.join(free)} encore disponibles."
        return f"🟢 {len(free)} place(s) disponible(s) : {', '.join(free)}. Occupation : {occ}%."

    def is_available(self) -> bool:
        return self._client is not None
