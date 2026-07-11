"""
Couche Base de Données - MongoDB
Séries temporelles de l'occupation des places
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo non disponible, stockage mémoire utilisé")


class DatabaseManager:
    """Gère la persistance MongoDB des données de capteurs."""

    def __init__(self, uri: str = "mongodb://localhost:27017/", db_name: str = "parking_iot"):
        self._uri     = uri
        self._db_name = db_name
        self._client  = None
        self._db      = None
        self._memory  = []          # fallback si MongoDB absent
        self._connect()

    def _connect(self):
        if not MONGO_AVAILABLE:
            logger.info("[DB] Mode mémoire (pymongo absent)")
            return
        try:
            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=2000)
            self._client.admin.command("ping")
            self._db = self._client[self._db_name]
            # Index TTL : conservation 7 jours
            self._db.sensor_records.create_index(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=7 * 24 * 3600
            )
            logger.info(f"[DB] MongoDB connecté : {self._db_name}")
        except Exception as e:
            logger.warning(f"[DB] MongoDB indisponible ({e}), mode mémoire")
            self._client = None

    def insert_sensor_record(self, record: dict):
        if self._db is not None:
            try:
                self._db.sensor_records.insert_one(record)
                return
            except Exception as e:
                logger.error(f"[DB] Erreur insert: {e}")
        # Fallback mémoire (max 10 000 entrées)
        self._memory.append(record)
        if len(self._memory) > 10_000:
            self._memory = self._memory[-10_000:]

    def get_records_since(self, since: datetime) -> list:
        if self._db is not None:
            try:
                cursor = self._db.sensor_records.find(
                    {"timestamp": {"$gte": since}},
                    sort=[("timestamp", ASCENDING)],
                    projection={"_id": 0}
                )
                return list(cursor)
            except Exception as e:
                logger.error(f"[DB] Erreur lecture: {e}")
        return [r for r in self._memory if r.get("timestamp", datetime.min) >= since]

    def get_daily_stats(self) -> dict:
        since = datetime.utcnow() - timedelta(hours=24)
        records = self.get_records_since(since)
        if not records:
            return {"avg_occupancy": 0, "max_occupancy": 0, "record_count": 0}

        rates = [r.get("occupancy_rate", 0) for r in records]
        return {
            "avg_occupancy": round(sum(rates) / len(rates) * 100, 1),
            "max_occupancy": round(max(rates) * 100, 1),
            "min_occupancy": round(min(rates) * 100, 1),
            "record_count":  len(records),
            "period":        "24h",
        }

    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False


# Singleton
db_manager = DatabaseManager()
