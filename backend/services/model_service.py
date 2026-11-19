import json
import os
from typing import Dict, List, Optional

import joblib

from backend.config import (
    EVALUATION_RESULTS_PATH,
    MODEL_1H_PATH,
    MODEL_24H_PATH,
    TRAINING_METADATA_PATH,
)


class ModelService:
    """
    Singleton service that loads the Phase 5 trained XGBoost models and
    evaluation artifacts exactly once at backend startup.
    Never retrains or overwrites saved models.
    """

    def __init__(self):
        self.model_1h_bundle: Optional[dict] = None
        self.model_24h_bundle: Optional[dict] = None
        self.training_metadata: Optional[dict] = None
        self.evaluation_results: Optional[dict] = None
        self.load_errors: List[str] = []
        self._loaded: bool = False

    def load_all(self) -> None:
        if self._loaded:
            return
        self.load_errors = []

        # 1. Load 1-hour-ahead XGBoost bundle
        if not os.path.exists(MODEL_1H_PATH):
            self.load_errors.append(f"Missing 1h model file: {MODEL_1H_PATH}")
        else:
            try:
                self.model_1h_bundle = joblib.load(MODEL_1H_PATH)
            except Exception as exc:
                self.load_errors.append(f"Failed loading 1h model: {exc}")

        # 2. Load 24-hour-ahead XGBoost bundle
        if not os.path.exists(MODEL_24H_PATH):
            self.load_errors.append(f"Missing 24h model file: {MODEL_24H_PATH}")
        else:
            try:
                self.model_24h_bundle = joblib.load(MODEL_24H_PATH)
            except Exception as exc:
                self.load_errors.append(f"Failed loading 24h model: {exc}")

        # 3. Load Phase 5 training metadata
        if not os.path.exists(TRAINING_METADATA_PATH):
            self.load_errors.append(f"Missing training metadata: {TRAINING_METADATA_PATH}")
        else:
            try:
                with open(TRAINING_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.training_metadata = json.load(f)
            except Exception as exc:
                self.load_errors.append(f"Failed loading training metadata: {exc}")

        # 4. Load Phase 5 evaluation results
        if not os.path.exists(EVALUATION_RESULTS_PATH):
            self.load_errors.append(f"Missing evaluation results: {EVALUATION_RESULTS_PATH}")
        else:
            try:
                with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
                    self.evaluation_results = json.load(f)
            except Exception as exc:
                self.load_errors.append(f"Failed loading evaluation results: {exc}")

        self._loaded = True

    def is_ready(self) -> bool:
        return (
            self._loaded
            and self.model_1h_bundle is not None
            and self.model_24h_bundle is not None
            and len(self.load_errors) == 0
        )

    def get_bundle(self, horizon: str) -> dict:
        if not self._loaded:
            self.load_all()
        if horizon == "1h":
            if self.model_1h_bundle is None:
                raise RuntimeError("1-hour-ahead model is not loaded.")
            return self.model_1h_bundle
        if horizon == "24h":
            if self.model_24h_bundle is None:
                raise RuntimeError("24-hour-ahead model is not loaded.")
            return self.model_24h_bundle
        raise ValueError(f"Unsupported horizon '{horizon}'. Expected '1h' or '24h'.")

    def get_seen_stations(self) -> set:
        if not self._loaded:
            self.load_all()
        if self.model_1h_bundle and "seen_stations" in self.model_1h_bundle:
            return set(self.model_1h_bundle["seen_stations"])
        return set()

    def get_health_info(self) -> Dict:
        if not self._loaded:
            self.load_all()
        return {
            "models_ready": self.is_ready(),
            "model_1h_loaded": self.model_1h_bundle is not None,
            "model_24h_loaded": self.model_24h_bundle is not None,
            "model_1h_info": {
                "model_name": self.model_1h_bundle["model_name"],
                "horizon": self.model_1h_bundle["horizon"],
                "feature_count": len(self.model_1h_bundle["feature_cols"]),
                "train_years": self.model_1h_bundle.get("train_years"),
                "seen_stations_count": len(self.model_1h_bundle.get("seen_stations", [])),
            }
            if self.model_1h_bundle
            else None,
            "model_24h_info": {
                "model_name": self.model_24h_bundle["model_name"],
                "horizon": self.model_24h_bundle["horizon"],
                "feature_count": len(self.model_24h_bundle["feature_cols"]),
                "train_years": self.model_24h_bundle.get("train_years"),
                "seen_stations_count": len(self.model_24h_bundle.get("seen_stations", [])),
            }
            if self.model_24h_bundle
            else None,
            "errors": self.load_errors,
        }


model_service = ModelService()
