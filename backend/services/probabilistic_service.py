"""
backend/services/probabilistic_service.py
Service layer for probabilistic arrival forecasts and distribution-propagated availability.
Loads probabilistic model artifacts and evaluation cache once at startup.
"""

import json
import os
from typing import Dict, List, Optional
import joblib

from backend.config import BASE_DIR
from ml.availability_prediction import availability_service

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "probabilistic_evaluation")

PROB_MODEL_1H_PATH = os.path.join(MODELS_DIR, "best_probabilistic_model_1h.joblib")
PROB_MODEL_24H_PATH = os.path.join(MODELS_DIR, "best_probabilistic_model_24h.joblib")
PROB_META_PATH = os.path.join(MODELS_DIR, "probabilistic_training_metadata.json")
EVAL_RESULTS_PATH = os.path.join(REPORTS_DIR, "evaluation_results.json")
OPERATIONAL_EVAL_PATH = os.path.join(REPORTS_DIR, "operational_erlang_b_evaluation.json")


class ProbabilisticService:
    """
    Singleton service managing probabilistic count forecasting and
    distribution-propagated Erlang-B availability queries.
    """

    def __init__(self):
        self.model_1h_bundle: Optional[dict] = None
        self.model_24h_bundle: Optional[dict] = None
        self.metadata: Optional[dict] = None
        self.evaluation_results: Optional[dict] = None
        self.operational_evaluation: Optional[dict] = None
        self._loaded: bool = False
        self.load_errors: List[str] = []

    def load_all(self) -> None:
        if self._loaded:
            return
        self.load_errors = []

        # 1. 1-hour probabilistic bundle
        if os.path.exists(PROB_MODEL_1H_PATH):
            try:
                self.model_1h_bundle = joblib.load(PROB_MODEL_1H_PATH)
            except Exception as exc:
                self.load_errors.append(f"Failed loading 1h prob model: {exc}")

        # 2. 24-hour probabilistic bundle
        if os.path.exists(PROB_MODEL_24H_PATH):
            try:
                self.model_24h_bundle = joblib.load(PROB_MODEL_24H_PATH)
            except Exception as exc:
                self.load_errors.append(f"Failed loading 24h prob model: {exc}")

        # 3. Training metadata
        if os.path.exists(PROB_META_PATH):
            try:
                with open(PROB_META_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as exc:
                self.load_errors.append(f"Failed loading prob metadata: {exc}")

        # 4. Calibration & proper scoring evaluation results
        if os.path.exists(EVAL_RESULTS_PATH):
            try:
                with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
                    self.evaluation_results = json.load(f)
            except Exception as exc:
                self.load_errors.append(f"Failed loading eval results: {exc}")

        # 5. Operational Erlang-B evaluation
        if os.path.exists(OPERATIONAL_EVAL_PATH):
            try:
                with open(OPERATIONAL_EVAL_PATH, "r", encoding="utf-8") as f:
                    self.operational_evaluation = json.load(f)
            except Exception as exc:
                self.load_errors.append(f"Failed loading operational eval: {exc}")

        self._loaded = True

    def get_arrivals(self, station_id: str, horizon: str = "1h") -> Dict:
        """Returns PMF, mean, P(0), P(>=1), P(>=2), quantiles, and plain language interpretation."""
        self.load_all()
        est = availability_service.estimate_station_probabilistic_availability(station_id, horizon=horizon)
        if "error" in est:
            return {"status": "error", "message": est["error"]}
        if est.get("status") == "insufficient_history":
            return est

        return {
            "status": "success",
            "station_id": est["station_id"],
            "short_name": est["short_name"],
            "horizon": est["horizon"],
            "target_timestamp_pt": est.get("target_timestamp_pt"),
            "target_hour": est.get("target_hour"),
            "observed_point_count": est["observed_point_count"],
            "arrival_distribution": est["arrival_distribution"],
            "availability_confidence": est["availability_confidence"],
            "data_limitations": est["data_limitations"],
        }

    def get_availability(self, station_id: str, horizon: str = "1h") -> Dict:
        """Returns point Erlang-B vs distribution-propagated Erlang-B with 5-95% uncertainty range."""
        self.load_all()
        est = availability_service.estimate_station_probabilistic_availability(station_id, horizon=horizon)
        if "error" in est:
            return {"status": "error", "message": est["error"]}
        if est.get("status") == "insufficient_history":
            return est

        return {
            "status": "success",
            "station_id": est["station_id"],
            "short_name": est["short_name"],
            "horizon": est["horizon"],
            "target_timestamp_pt": est.get("target_timestamp_pt"),
            "target_hour": est.get("target_hour"),
            "observed_point_count": est["observed_point_count"],
            "predicted_session_arrivals": est["predicted_session_arrivals"],
            "mean_plugged_dwell_hours": est["mean_plugged_dwell_hours"],
            "prior_plugged_in_vehicles": est["prior_plugged_in_vehicles"],
            "arrival_summary": {
                "mean": est["arrival_distribution"]["mean"],
                "p_zero": est["arrival_distribution"]["p_zero"],
                "p_ge_1": est["arrival_distribution"]["p_ge_1"],
                "p_ge_2": est["arrival_distribution"]["p_ge_2"],
                "plain_language": est["arrival_distribution"]["plain_language"],
            },
            "erlang_b_availability": est["erlang_b_availability"],
            "availability_confidence": est["availability_confidence"],
            "data_limitations": est["data_limitations"],
        }

    def get_calibration(self) -> Dict:
        """Returns cached scoring rules, PIT histogram bins, reliability diagram points, and coverage."""
        self.load_all()
        if self.evaluation_results is None:
            return {"status": "error", "message": "Probabilistic evaluation results are not available."}

        return {
            "status": "success",
            "evaluation_results": self.evaluation_results,
            "operational_evaluation": self.operational_evaluation,
        }

    def get_summary(self) -> Dict:
        """Returns headline metrics for the dashboard overview."""
        self.load_all()
        ev = self.evaluation_results
        op = self.operational_evaluation or {}

        test_19 = ev.get("test_2019", {}) if ev else {}
        ood_20 = ev.get("ood_2020", {}) if ev else {}

        # Best model results
        best_key = "xgboost_as_poisson" if "xgboost_as_poisson" in test_19 else "poisson_glm"
        best_19 = test_19.get(best_key, {})
        emp_19 = test_19.get("empirical_baseline", {})

        return {
            "status": "success",
            "best_model_name": "XGBoost-as-Poisson" if best_key == "xgboost_as_poisson" else "Poisson GLM",
            "test_2019_headline": {
                "log_score": best_19.get("log_score"),
                "rps": best_19.get("rps"),
                "zero_brier": best_19.get("zero_brier"),
                "roc_auc_busy": best_19.get("roc_auc_busy"),
                "pr_auc_busy": best_19.get("pr_auc_busy"),
                "coverage_90": best_19.get("coverage_90"),
                "pit_ks_stat": best_19.get("pit_ks_stat"),
                "ece_ge_1": best_19.get("ece_ge_1"),
            },
            "baseline_comparison": {
                "baseline_name": "Empirical Hour-of-Week",
                "baseline_log_score": emp_19.get("log_score"),
                "baseline_rps": emp_19.get("rps"),
                "baseline_zero_brier": emp_19.get("zero_brier"),
                "log_score_reduction_pct": round(
                    ((emp_19.get("log_score", 0.4125) - best_19.get("log_score", 0.3523)) / emp_19.get("log_score", 0.4125)) * 100.0, 1
                ),
                "rps_reduction_pct": round(
                    ((emp_19.get("rps", 0.129) - best_19.get("rps", 0.1099)) / emp_19.get("rps", 0.129)) * 100.0, 1
                ),
            },
            "operational_erlang_b": {
                "point_brier": op.get("point_erlang_brier_score"),
                "distribution_brier": op.get("distribution_erlang_brier_score"),
                "brier_improvement_percent": op.get("brier_improvement_percent"),
                "evaluated_hours": op.get("n_evaluated_hours"),
                "full_station_hours": op.get("full_station_hours"),
            },
            "temporal_shift_2020": {
                "best_log_score_2020": ood_20.get(best_key, {}).get("log_score"),
                "baseline_log_score_2020": ood_20.get("empirical_baseline", {}).get("log_score"),
                "baseline_ece_ge_1_2020": ood_20.get("empirical_baseline", {}).get("ece_ge_1"),
            },
        }


probabilistic_service = ProbabilisticService()
