"""
ml/availability_prediction.py
Future charging-point availability estimation engine.
Integrates Phase 5 XGBoost demand models (1h & 24h), observed charging point inventory,
and historical plugged-in dwell duration distributions via an Erlang loss / survival queueing model.
Returns expected occupied points, estimated free observed points, probability of zero free points,
confidence rating, and methodological data limitations.
Never returns negative free points or values exceeding validated observed capacity.
"""

import math
from typing import Dict, List, Optional
import numpy as np

from backend.services.data_service import data_service
from backend.services.model_service import model_service
from backend.services.prediction_service import prediction_service
from ml.charging_point_inventory import inventory_service


def compute_erlang_b(servers: int, offered_load: float) -> float:
    """
    Computes Erlang-B blocking probability for c servers and offered traffic load a = lambda / mu.
    Uses stable backward recurrence:
    1 / B(k, a) = 1 + (k / a) * (1 / B(k-1, a))
    """
    if servers <= 0:
        return 1.0
    if offered_load <= 1e-9:
        return 0.0

    inv_b = 1.0
    for k in range(1, servers + 1):
        inv_b = 1.0 + (k / offered_load) * inv_b
    return float(np.clip(1.0 / inv_b, 0.0, 1.0))


class AvailabilityPredictionService:
    """
    Predicts future charging-point availability for a given station and target horizon/timestamp.
    Integrates pre-trained Phase 5 XGBoost arrival predictions with observed hardware inventory.
    """

    def estimate_station_availability_1h(self, station_id: str) -> Dict:
        """
        Estimates charging-point availability for target hour T_last + 1h.
        """
        # Fetch station metadata and inventory
        st_meta = data_service.get_station_detail(station_id)
        if st_meta is None:
            return {"error": f"Unknown station_id: '{station_id}'"}

        st_inv = inventory_service.get_station_summary(station_id)
        obs_points = st_inv["active_observed_points_count"] if st_inv else 2

        # 1-Hour-Ahead demand forecast
        pred_1h = prediction_service.predict_station_1h(station_id, persist=False)
        if pred_1h.get("status") == "insufficient_history":
            return {
                "status": "insufficient_history",
                "station_id": station_id,
                "short_name": st_meta["short_name"],
                "horizon": "1-hour",
                "target_timestamp_pt": st_meta["last_hour_pt"],
                "observed_point_count": obs_points,
                "expected_occupied_points": None,
                "estimated_free_points": None,
                "estimated_free_points_rounded": None,
                "probability_zero_free_points": None,
                "availability_confidence": "Low / Insufficient History",
                "warning": pred_1h.get("reason"),
                "data_limitations": (
                    "Station has fewer than 24 observed historical hours. "
                    "Availability estimates are withheld to avoid false precision."
                ),
            }

        arr_rate = float(pred_1h["predicted_sessions"])
        dwell_min = float(st_meta["mean_plugged_min"] or 157.0)
        dwell_hours = max(dwell_min / 60.0, 0.5)
        service_rate = 1.0 / dwell_hours
        offered_load = arr_rate * dwell_hours

        # Existing vehicles carryover
        prior_plugged = int(pred_1h["prior_plugged_in_vehicles"])
        carryover_prob = math.exp(-service_rate)
        expected_carryover = min(float(prior_plugged) * carryover_prob, float(obs_points))

        # Erlang-B blocking probability for new arrivals
        p_blocked = compute_erlang_b(obs_points, offered_load)
        expected_new_occ = offered_load * (1.0 - p_blocked)

        # Expected total occupied (clamped to [0, obs_points])
        exp_occ = min(expected_carryover + expected_new_occ, float(obs_points))
        exp_occ = max(exp_occ, 0.0)

        # Estimated free observed points (strictly in [0, obs_points])
        est_free = max(0.0, float(obs_points) - exp_occ)
        est_free_rounded = int(round(est_free))

        # Overall probability of zero free points
        # If expected occupied approaches capacity, P_zero approaches P_blocked + carryover influence
        p_zero_free = min(1.0, max(p_blocked, (exp_occ / float(obs_points)) ** float(obs_points)))

        confidence = "High" if not st_meta["is_cold_start"] and st_meta["active_days"] > 180 else "Medium"
        if st_meta["is_cold_start"]:
            confidence = "Low / Cold-Start"

        return {
            "status": "ok",
            "station_id": station_id,
            "short_name": st_meta["short_name"],
            "horizon": "1-hour",
            "target_timestamp_pt": pred_1h["target_timestamp_pt"],
            "target_hour": pred_1h["target_hour"],
            "observed_point_count": obs_points,
            "predicted_session_arrivals": arr_rate,
            "expected_occupied_points": round(exp_occ, 2),
            "estimated_free_points": round(est_free, 2),
            "estimated_free_points_rounded": est_free_rounded,
            "probability_zero_free_points": round(p_zero_free, 3),
            "availability_status": (
                "Likely Available" if est_free >= 1.0 else ("Limited Availability" if est_free >= 0.3 else "High Occupancy Risk")
            ),
            "mean_plugged_dwell_hours": round(dwell_hours, 2),
            "prior_plugged_in_vehicles": prior_plugged,
            "availability_confidence": confidence,
            "is_cold_start": st_meta["is_cold_start"],
            "data_limitations": (
                "Availability reflects observed charging-point telemetry and expected arrival rate. "
                "Does not account for unrecorded offline chargers, out-of-order equipment, or non-charging vehicle blocking."
            ),
        }

    def estimate_station_availability_24h(self, station_id: str) -> Dict:
        """
        Estimates hourly charging-point availability across all 24 day-ahead hours.
        """
        st_meta = data_service.get_station_detail(station_id)
        if st_meta is None:
            return {"error": f"Unknown station_id: '{station_id}'"}

        st_inv = inventory_service.get_station_summary(station_id)
        obs_points = st_inv["active_observed_points_count"] if st_inv else 2

        pred_24h = prediction_service.predict_station_24h(station_id, persist=False)
        if pred_24h.get("status") == "insufficient_history":
            return {
                "status": "insufficient_history",
                "station_id": station_id,
                "short_name": st_meta["short_name"],
                "horizon": "24-hour",
                "hourly_availability": [],
                "observed_point_count": obs_points,
                "availability_confidence": "Low / Insufficient History",
                "warning": pred_24h.get("reason"),
            }

        dwell_min = float(st_meta["mean_plugged_min"] or 157.0)
        dwell_hours = max(dwell_min / 60.0, 0.5)

        hourly_results = []
        for h_pred in pred_24h["hourly_predictions"]:
            arr_rate = float(h_pred["predicted_sessions"])
            offered_load = arr_rate * dwell_hours
            p_blocked = compute_erlang_b(obs_points, offered_load)
            exp_occ = min(offered_load * (1.0 - p_blocked), float(obs_points))
            est_free = max(0.0, float(obs_points) - exp_occ)
            est_free_rounded = int(round(est_free))
            p_zero_free = min(1.0, max(p_blocked, (exp_occ / float(obs_points)) ** float(obs_points)))

            hourly_results.append(
                {
                    "step_ahead": h_pred["step_ahead"],
                    "target_timestamp_pt": h_pred["target_timestamp_pt"],
                    "hour": h_pred["hour"],
                    "label": h_pred["label"],
                    "predicted_session_arrivals": arr_rate,
                    "expected_occupied_points": round(exp_occ, 2),
                    "estimated_free_points": round(est_free, 2),
                    "estimated_free_points_rounded": est_free_rounded,
                    "probability_zero_free_points": round(p_zero_free, 3),
                    "status_label": "Likely Free" if est_free >= 1.0 else ("Limited" if est_free >= 0.3 else "Crowded"),
                }
            )

        mean_free = float(np.mean([h["estimated_free_points"] for h in hourly_results]))
        min_free_item = min(hourly_results, key=lambda x: x["estimated_free_points"])

        confidence = "High" if not st_meta["is_cold_start"] and st_meta["active_days"] > 180 else "Medium"
        if st_meta["is_cold_start"]:
            confidence = "Low / Cold-Start"

        return {
            "status": "ok",
            "station_id": station_id,
            "short_name": st_meta["short_name"],
            "horizon": "24-hour",
            "observed_point_count": obs_points,
            "mean_estimated_free_points_24h": round(mean_free, 2),
            "tightest_availability_hour_pt": min_free_item["target_timestamp_pt"],
            "tightest_availability_hour": min_free_item["hour"],
            "tightest_estimated_free_points": min_free_item["estimated_free_points"],
            "highest_risk_of_full_station": min_free_item["probability_zero_free_points"],
            "availability_confidence": confidence,
            "hourly_availability": hourly_results,
            "data_limitations": (
                "Day-ahead estimates reflect Poisson expected arrivals and empirical dwell times. "
                "Actual availability may vary due to parking turnover and driver behavior."
            ),
        }

    def estimate_at_target_datetime(self, station_id: str, target_iso_pt: str) -> Dict:
        """
        Estimates availability at an arbitrary user-selected future timestamp.
        Interpolates from the 24-hour forecast profile if within 24h, or station's historical same-hour pattern.
        """
        st_meta = data_service.get_station_detail(station_id)
        if st_meta is None:
            return {"error": f"Unknown station_id: '{station_id}'"}

        st_inv = inventory_service.get_station_summary(station_id)
        obs_points = st_inv["active_observed_points_count"] if st_inv else 2

        # Check 24h forecast first
        pred_24 = self.estimate_station_availability_24h(station_id)
        if pred_24.get("status") == "ok":
            for h in pred_24["hourly_availability"]:
                if h["target_timestamp_pt"].startswith(target_iso_pt[:13]):
                    return {
                        "status": "ok",
                        "station_id": station_id,
                        "short_name": st_meta["short_name"],
                        "target_timestamp_pt": target_iso_pt,
                        "target_hour": h["hour"],
                        "observed_point_count": obs_points,
                        "predicted_session_arrivals": h["predicted_session_arrivals"],
                        "expected_occupied_points": h["expected_occupied_points"],
                        "estimated_free_points": h["estimated_free_points"],
                        "estimated_free_points_rounded": h["estimated_free_points_rounded"],
                        "probability_zero_free_points": h["probability_zero_free_points"],
                        "availability_status": h["status_label"],
                        "source": "24h_ahead_xgboost",
                        "availability_confidence": pred_24["availability_confidence"],
                        "data_limitations": pred_24["data_limitations"],
                    }

        # Fallback to historical clock hour profile
        try:
            target_dt = pd.to_datetime(target_iso_pt)
            target_hr = target_dt.hour
        except Exception:
            target_hr = 11

        hist_mean = float(st_meta["hourly_mean_sessions"][target_hr])
        dwell_min = float(st_meta["mean_plugged_min"] or 157.0)
        dwell_hours = max(dwell_min / 60.0, 0.5)
        offered_load = hist_mean * dwell_hours

        p_blocked = compute_erlang_b(obs_points, offered_load)
        exp_occ = min(offered_load * (1.0 - p_blocked), float(obs_points))
        est_free = max(0.0, float(obs_points) - exp_occ)

        return {
            "status": "ok",
            "station_id": station_id,
            "short_name": st_meta["short_name"],
            "target_timestamp_pt": target_iso_pt,
            "target_hour": target_hr,
            "observed_point_count": obs_points,
            "predicted_session_arrivals": round(hist_mean, 4),
            "expected_occupied_points": round(exp_occ, 2),
            "estimated_free_points": round(est_free, 2),
            "estimated_free_points_rounded": int(round(est_free)),
            "probability_zero_free_points": round(p_blocked, 3),
            "availability_status": "Likely Free" if est_free >= 1.0 else ("Limited" if est_free >= 0.3 else "Crowded"),
            "source": "historical_same_hour_baseline",
            "availability_confidence": "Medium",
            "data_limitations": (
                "Estimate derived from historical station-hour baseline. "
                "Actual availability depends on real-time vehicle arrivals."
            ),
        }


availability_service = AvailabilityPredictionService()

if __name__ == "__main__":
    res = availability_service.estimate_station_availability_1h("PALO ALTO CA / HAMILTON #2")
    print("1-Hour Availability Estimate:")
    print(res)
