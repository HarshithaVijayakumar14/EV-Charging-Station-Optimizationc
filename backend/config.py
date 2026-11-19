import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Upstream Phase 1-5 read-only artifacts (NEVER modified by the web app)
RAW_CSV = os.path.join(BASE_DIR, "EVChargingStationUsage.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "dataset", "processed", "ev_charging_cleaned.csv")
HOURLY_FEATURES_CSV = os.path.join(BASE_DIR, "dataset", "processed", "ev_charging_hourly_features.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_1H_PATH = os.path.join(MODELS_DIR, "best_model_1h.joblib")
MODEL_24H_PATH = os.path.join(MODELS_DIR, "best_model_24h.joblib")
TRAINING_METADATA_PATH = os.path.join(MODELS_DIR, "training_metadata.json")
EVALUATION_RESULTS_PATH = os.path.join(BASE_DIR, "reports", "model_evaluation", "evaluation_results.json")

# Backend-owned storage (derived read-only cache + SQLite app DB)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
CACHE_DIR = os.path.join(BACKEND_DIR, "cache")
DB_PATH = os.path.join(BACKEND_DIR, "database", "ev_app.db")

# Flask settings
HOST = os.environ.get("EV_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("EV_BACKEND_PORT", "5000"))
DEBUG = os.environ.get("EV_BACKEND_DEBUG", "false").lower() == "true"

# Minimum historical span required to produce reliable lag-based predictions
MIN_HISTORY_HOURS_FOR_PREDICTION = 24
