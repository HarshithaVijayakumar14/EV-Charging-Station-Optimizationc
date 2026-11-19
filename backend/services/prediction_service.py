from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import MIN_HISTORY_HOURS_FOR_PREDICTION
from backend.database.db import log_predictions
from backend.services.data_service import data_service
from backend.services.model_service import model_service
from ml.common import (
    FEATURES_1H,
    FEATURES_24H,
    add_24h_ahead_features,
    get_us_federal_holidays,
)
from ml.predict_demand import load_model_bundle  # reused via model_service / parity


def _us_pacific_offset_hours(utc_ts: pd.Timestamp) -> int:
    y = int(utc_ts.year)
    mar1 = datetime(y, 3, 1)
    first_sun_mar = 1 + (6 - mar1.weekday()) % 7
    dst_start = pd.Timestamp(datetime(y, 3, first_sun_mar + 7, 10, 0))
    nov1 = datetime(y, 11, 1)
    first_sun_nov = 1 + (6 - nov1.weekday()) % 7
    dst_end = pd.Timestamp(datetime(y, 11, first_sun_nov, 9, 0))
    return -7 if dst_start <= utc_ts < dst_end else -8


def _calendar_features_for_utc(utc_ts: pd.Timestamp) -> dict:
    off = _us_pacific_offset_hours(utc_ts)
    pt_ts = utc_ts + pd.Timedelta(hours=off)
    off_str = "-07:00" if off == -7 else "-08:00"
    pt_iso = pt_ts.strftime("%Y-%m-%d %H:%M:%S") + off_str

    hour = int(pt_ts.hour)
    day = int(pt_ts.day)
    dow = int(pt_ts.dayofweek)
    month = int(pt_ts.month)
    year = int(pt_ts.year)
    is_wknd = 1 if dow >= 5 else 0
    holidays = get_us_federal_holidays([year])
    is_hol = 1 if pt_ts.date() in holidays else 0
    is_work = 1 if (is_wknd == 0 and is_hol == 0) else 0
    is_paid = 1 if pt_ts >= pd.Timestamp("2017-08-01 00:00:00") else 0

    return {
        "hour_timestamp_pt": pt_iso,
        "hour_timestamp_utc": utc_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": hour,
        "day": day,
        "day_of_week": dow,
        "month": month,
        "year": year,
        "is_weekend": is_wknd,
        "is_us_federal_holiday": is_hol,
        "is_working_day": is_work,
        "hour_sin": round(float(np.sin(2.0 * np.pi * hour / 24.0)), 6),
        "hour_cos": round(float(np.cos(2.0 * np.pi * hour / 24.0)), 6),
        "dow_sin": round(float(np.sin(2.0 * np.pi * dow / 7.0)), 6),
        "dow_cos": round(float(np.cos(2.0 * np.pi * dow / 7.0)), 6),
        "month_sin": round(float(np.sin(2.0 * np.pi * (month - 1) / 12.0)), 6),
        "month_cos": round(float(np.cos(2.0 * np.pi * (month - 1) / 12.0)), 6),
        "is_paid_era": is_paid,
    }


def _score_with_loaded_bundle(df_features: pd.DataFrame, horizon: str) -> np.ndarray:
    """
    Score a feature DataFrame using the singleton pre-loaded Phase 5 XGBoost bundle
    following the exact same transformation and non-negative clipping as ml/predict_demand.py.
    """
    bundle = model_service.get_bundle(horizon)
    prep = bundle["preprocessor"]
    impute = bool(bundle.get("impute_missing", False))
    X = prep.transform(df_features, impute=impute)
    preds = bundle["model"].predict(X)
    return np.clip(np.asarray(preds, dtype=np.float64), 0.0, None)


class PredictionService:
    """
    Leakage-free 1-hour-ahead and 24-hour-ahead station demand forecasting service.
    - Never caches predictions across different stations or forecast origins.
    - Uses ONLY historical observations strictly <= forecast_origin (T_last for next-step forecasts,
      and strictly <= t - 24h for every hour in the 24-hour-ahead horizon).
    - Clearly separates forecast predictions from historical Actual-vs-Predicted backtest windows.
    """

    def _build_next_1h_feature_row(self, station_id: str) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Build the exact causal feature row for target hour T_last + 1h using ONLY
        actual historical observations up to T_last.
        """
        meta = data_service.get_station_detail(station_id)
        panel = data_service.get_station_recent_panel(station_id)
        if meta is None or panel is None or len(panel) == 0:
            return None, {"error": f"Unknown station_id: '{station_id}'"}

        if not meta["has_sufficient_history"]:
            return None, {
                "status": "insufficient_history",
                "station_id": station_id,
                "short_name": meta["short_name"],
                "horizon": "1-hour",
                "forecast_origin_pt": meta["last_hour_pt"],
                "predicted_sessions": None,
                "predicted_sessions_rounded": None,
                "is_cold_start": meta["is_cold_start"],
                "span_hours": meta["span_hours"],
                "reason": (
                    f"Insufficient historical data ({meta['span_hours']} hour(s) available; "
                    f"minimum {MIN_HISTORY_HOURS_FOR_PREDICTION} hours required for causal lag features)."
                ),
            }

        last_row = panel.iloc[-1]
        last_utc = pd.Timestamp(last_row["hour_timestamp_utc"])
        target_utc = last_utc + pd.Timedelta(hours=1)
        cal = _calendar_features_for_utc(target_utc)

        s_arr = panel["sessions_count"].to_numpy(dtype=np.float64)
        e_arr = panel["energy_kwh_arrival_sum"].to_numpy(dtype=np.float64)
        n_hist = len(s_arr)

        def lag_val(arr: np.ndarray, k: int) -> float:
            return float(arr[-k]) if n_hist >= k else float("nan")

        # Rolling strictly over [T_last - W + 1 .. T_last] (= [target - W .. target - 1])
        roll24_s_mean = float(np.mean(s_arr[-24:])) if n_hist >= 24 else float("nan")
        roll24_s_sum = float(np.sum(s_arr[-24:])) if n_hist >= 24 else float("nan")
        roll168_s_mean = float(np.mean(s_arr[-168:])) if n_hist >= 168 else float("nan")
        roll168_s_sum = float(np.sum(s_arr[-168:])) if n_hist >= 168 else float("nan")
        roll24_e_mean = float(np.mean(e_arr[-24:])) if n_hist >= 24 else float("nan")
        roll168_e_mean = float(np.mean(e_arr[-168:])) if n_hist >= 168 else float("nan")

        # Hours since last non-zero session strictly before target_utc (<= T_last)
        nz_idx = np.where(s_arr > 0)[0]
        if len(nz_idx) > 0:
            hours_since_last = float(n_hist - int(nz_idx[-1]))
        else:
            prev_hsl = last_row.get("hours_since_last_session", np.nan)
            hours_since_last = float(prev_hsl) + 1.0 if pd.notna(prev_hsl) else float("nan")

        # Expanding station-hour mean up to T_last for target clock hour
        target_hr = cal["hour"]
        st_hr_mean = float(meta["hourly_mean_sessions"][target_hr])

        row = {
            "station_id": station_id,
            **cal,
            "station_age_days": round(float(last_row["station_age_days"]) + 1.0 / 24.0, 4),
            "station_cum_sessions_prior": int(meta["total_sessions"]),
            "station_expanding_mean_sessions": float(meta["overall_hourly_mean"]),
            "station_hour_expanding_mean_sessions": st_hr_mean,
            "hours_since_last_session": hours_since_last,
            "station_hist_mean_energy_per_session": float(meta["mean_energy_kwh"]),
            "station_hist_mean_plugged_dur_min": float(meta["mean_plugged_min"]) if meta["mean_plugged_min"] is not None else float("nan"),
            "station_hist_mean_active_chg_min": float(meta["mean_charging_min"]) if meta["mean_charging_min"] is not None else float("nan"),
            "prior_plugged_in_vehicles_at_hour_start": int(meta["prior_plugged_at_next_hour"]),
            "prior_active_charging_vehicles_at_hour_start": int(meta["prior_active_at_next_hour"]),
            "lag_1h_sessions": lag_val(s_arr, 1),
            "lag_2h_sessions": lag_val(s_arr, 2),
            "lag_3h_sessions": lag_val(s_arr, 3),
            "lag_24h_sessions": lag_val(s_arr, 24),
            "lag_168h_sessions": lag_val(s_arr, 168),
            "rolling_mean_24h_sessions": roll24_s_mean,
            "rolling_sum_24h_sessions": roll24_s_sum,
            "rolling_mean_168h_sessions": roll168_s_mean,
            "rolling_sum_168h_sessions": roll168_s_sum,
            "lag_1h_energy_kwh": lag_val(e_arr, 1),
            "lag_24h_energy_kwh": lag_val(e_arr, 24),
            "lag_168h_energy_kwh": lag_val(e_arr, 168),
            "rolling_mean_24h_energy_kwh": roll24_e_mean,
            "rolling_mean_168h_energy_kwh": roll168_e_mean,
        }
        return row, None

    def _build_next_24h_feature_rows(self, station_id: str) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """
        Build 24 strictly causal feature rows for target hours T_last + 1h .. T_last + 24h.
        Every lag & rolling feature for target hour t_k = T_last + k uses ONLY actual data <= t_k - 24h <= T_last!
        Zero future observations or recursive fake predictions are used.
        """
        meta = data_service.get_station_detail(station_id)
        panel = data_service.get_station_recent_panel(station_id)
        if meta is None or panel is None or len(panel) == 0:
            return None, {"error": f"Unknown station_id: '{station_id}'"}

        if not meta["has_sufficient_history"]:
            return None, {
                "status": "insufficient_history",
                "station_id": station_id,
                "short_name": meta["short_name"],
                "horizon": "24-hour",
                "forecast_origin_pt": meta["last_hour_pt"],
                "hourly_predictions": [],
                "total_predicted_sessions_24h": None,
                "is_cold_start": meta["is_cold_start"],
                "span_hours": meta["span_hours"],
                "reason": (
                    f"Insufficient historical data ({meta['span_hours']} hour(s) available; "
                    f"minimum {MIN_HISTORY_HOURS_FOR_PREDICTION} hours required for 24-hour-ahead causal lags)."
                ),
            }

        last_row = panel.iloc[-1]
        last_utc = pd.Timestamp(last_row["hour_timestamp_utc"])
        last_age = float(last_row["station_age_days"])

        s_arr = panel["sessions_count"].to_numpy(dtype=np.float64)
        e_arr = panel["energy_kwh_arrival_sum"].to_numpy(dtype=np.float64)
        n_hist = len(s_arr)  # index n_hist - 1 corresponds to T_last

        rows = []
        for k in range(1, 25):
            target_utc = last_utc + pd.Timedelta(hours=k)
            cal = _calendar_features_for_utc(target_utc)
            # Forecast cutoff for 24h-ahead target t_k = T_last + k is (t_k - 24h) = T_last - (24 - k)
            # In panel array indices: idx_t24 = (n_hist - 1) - (24 - k) = n_hist - 25 + k
            idx_t24 = n_hist - 25 + k  # always <= n_hist - 1 (strictly observed <= T_last!)

            def val_at_lag_from_target(lag_h: int, arr: np.ndarray) -> float:
                # target is at conceptual index (n_hist - 1 + k); lag_h before target is (n_hist - 1 + k - lag_h)
                idx = n_hist - 1 + k - lag_h
                return float(arr[idx]) if 0 <= idx < n_hist else float("nan")

            def roll_ending_at_t24(win: int, arr: np.ndarray, fn) -> float:
                start_i = idx_t24 - win + 1
                if start_i < 0 or idx_t24 >= n_hist:
                    return float("nan")
                return float(fn(arr[start_i : idx_t24 + 1]))

            # Cumulative sessions & expanding mean strictly up to idx_t24 (<= T_last)
            tail_excluded_sessions = float(np.sum(s_arr[idx_t24 + 1 :])) if idx_t24 + 1 < n_hist else 0.0
            tail_excluded_hours = (n_hist - 1) - idx_t24
            cum_at_t24 = max(0.0, float(meta["total_sessions"]) - tail_excluded_sessions)
            hours_at_t24 = max(1, int(meta["span_hours"]) - tail_excluded_hours)
            exp_mean_at_t24 = cum_at_t24 / hours_at_t24

            rows.append(
                {
                    "station_id": station_id,
                    **cal,
                    "station_age_days": round(last_age + k / 24.0, 4),
                    "station_cum_sessions_at_t24": cum_at_t24,
                    "station_expanding_mean_at_t24_sessions": exp_mean_at_t24,
                    # Same hour on strictly prior days (at target_utc, same clock hour yesterday is T_last + k - 24 <= T_last)
                    "station_hour_expanding_mean_sessions": float(meta["hourly_mean_sessions"][cal["hour"]]),
                    "lag_24h_sessions": val_at_lag_from_target(24, s_arr),
                    "lag_48h_sessions": val_at_lag_from_target(48, s_arr),
                    "lag_168h_sessions": val_at_lag_from_target(168, s_arr),
                    "lag_336h_sessions": val_at_lag_from_target(336, s_arr),
                    "rolling_mean_24h_at_t24_sessions": roll_ending_at_t24(24, s_arr, np.mean),
                    "rolling_sum_24h_at_t24_sessions": roll_ending_at_t24(24, s_arr, np.sum),
                    "rolling_mean_168h_at_t24_sessions": roll_ending_at_t24(168, s_arr, np.mean),
                    "rolling_sum_168h_at_t24_sessions": roll_ending_at_t24(168, s_arr, np.sum),
                    "lag_24h_energy_kwh": val_at_lag_from_target(24, e_arr),
                    "lag_168h_energy_kwh": val_at_lag_from_target(168, e_arr),
                    "rolling_mean_168h_at_t24_energy_kwh": roll_ending_at_t24(168, e_arr, np.mean),
                }
            )

        return pd.DataFrame(rows), None

    def predict_station_1h(self, station_id: str, persist: bool = True) -> dict:
        row, err = self._build_next_1h_feature_row(station_id)
        if err is not None:
            return err

        meta = data_service.get_station_detail(station_id)
        bundle = model_service.get_bundle("1h")
        df_1 = pd.DataFrame([row])
        pred_val = float(_score_with_loaded_bundle(df_1, horizon="1h")[0])

        result = {
            "status": "ok",
            "station_id": station_id,
            "short_name": meta["short_name"],
            "horizon": "1-hour",
            "metric_name": "predicted_session_arrivals",
            "metric_note": "Expected number of new charging sessions arriving during the target hour (not simultaneous occupancy or port capacity).",
            "model_name": bundle["model_name"],
            "forecast_origin_pt": meta["last_hour_pt"],
            "target_timestamp_pt": row["hour_timestamp_pt"],
            "target_hour": row["hour"],
            "predicted_sessions": round(pred_val, 4),
            "predicted_sessions_rounded": int(round(pred_val)),
            "historical_hour_avg": round(float(meta["hourly_mean_sessions"][row["hour"]]), 4),
            "overall_hourly_mean": meta["overall_hourly_mean"],
            "is_cold_start": meta["is_cold_start"],
            "cold_start_note": meta["cold_start_note"],
            "prior_plugged_in_vehicles": row["prior_plugged_in_vehicles_at_hour_start"],
            "prior_active_charging_vehicles": row["prior_active_charging_vehicles_at_hour_start"],
        }
        if persist:
            log_predictions([result])
        return result

    def predict_station_24h(self, station_id: str, persist: bool = True) -> dict:
        df_24, err = self._build_next_24h_feature_rows(station_id)
        if err is not None:
            return err

        meta = data_service.get_station_detail(station_id)
        bundle = model_service.get_bundle("24h")
        preds = _score_with_loaded_bundle(df_24, horizon="24h")

        hourly_items = []
        for idx, pred_val in enumerate(preds):
            r = df_24.iloc[idx]
            hr = int(r["hour"])
            p_f = float(pred_val)
            hourly_items.append(
                {
                    "step_ahead": idx + 1,
                    "target_timestamp_pt": str(r["hour_timestamp_pt"]),
                    "hour": hr,
                    "label": f"{hr:02d}:00",
                    "predicted_sessions": round(p_f, 4),
                    "predicted_sessions_rounded": int(round(p_f)),
                    "historical_hour_avg": round(float(meta["hourly_mean_sessions"][hr]), 4),
                }
            )

        total_24h = round(float(np.sum(preds)), 4)
        peak_item = max(hourly_items, key=lambda x: x["predicted_sessions"])
        # 24h-ahead target timestamp summary (step 24 = exact t + 24h)
        step24 = hourly_items[-1]

        result = {
            "status": "ok",
            "station_id": station_id,
            "short_name": meta["short_name"],
            "horizon": "24-hour",
            "metric_name": "predicted_session_arrivals",
            "metric_note": "Strictly <= t-24h day-ahead expected session arrivals across the next 24 clock hours (not simultaneous occupancy).",
            "model_name": bundle["model_name"],
            "forecast_origin_pt": meta["last_hour_pt"],
            "forecast_window_start_pt": hourly_items[0]["target_timestamp_pt"],
            "forecast_window_end_pt": step24["target_timestamp_pt"],
            "total_predicted_sessions_24h": total_24h,
            "total_predicted_sessions_24h_rounded": int(round(total_24h)),
            "mean_hourly_predicted_sessions": round(total_24h / 24.0, 4),
            "peak_predicted_hour_pt": peak_item["target_timestamp_pt"],
            "peak_predicted_hour": peak_item["hour"],
            "peak_predicted_sessions": peak_item["predicted_sessions"],
            "is_cold_start": meta["is_cold_start"],
            "cold_start_note": meta["cold_start_note"],
            "hourly_predictions": hourly_items,
        }
        if persist:
            log_predictions(
                [
                    {
                        "station_id": station_id,
                        "horizon": "24-hour",
                        "forecast_origin_pt": meta["last_hour_pt"],
                        "target_timestamp_pt": peak_item["target_timestamp_pt"],
                        "predicted_sessions": total_24h,
                        "predicted_sessions_rounded": int(round(total_24h)),
                        "model_name": bundle["model_name"],
                        "is_cold_start": meta["is_cold_start"],
                    }
                ]
            )
        return result

    def get_station_historical_backtest(self, station_id: str, hours: int = 72) -> dict:
        """
        Evaluate the trained 1h and 24h models on the station's most recent observed historical hours
        where actual `sessions_count` is known (strictly using causal Phase 4 features — zero leakage).
        Clearly marked as historical backtest, never presented as future predictions.
        """
        meta = data_service.get_station_detail(station_id)
        panel = data_service.get_station_recent_panel(station_id)
        if meta is None or panel is None:
            return {"error": f"Unknown station_id: '{station_id}'"}

        if not meta["has_sufficient_history"]:
            return {
                "status": "insufficient_history",
                "station_id": station_id,
                "records": [],
                "reason": meta["cold_start_note"],
            }

        # Add strictly <= t-24h features on the station's 504h panel before slicing the tail
        panel_24 = add_24h_ahead_features(panel)
        tail = panel_24.tail(min(int(hours), len(panel_24))).reset_index(drop=True)

        p1 = _score_with_loaded_bundle(tail, horizon="1h")
        p24 = _score_with_loaded_bundle(tail, horizon="24h")
        actuals = tail["sessions_count"].to_numpy(dtype=np.float64)

        records = [
            {
                "timestamp_pt": str(tail.loc[i, "hour_timestamp_pt"]),
                "hour": int(tail.loc[i, "hour"]),
                "actual_sessions": int(actuals[i]),
                "predicted_1h_sessions": round(float(p1[i]), 4),
                "predicted_24h_sessions": round(float(p24[i]), 4),
            }
            for i in range(len(tail))
        ]
        mae_1h = round(float(np.mean(np.abs(actuals - p1))), 4)
        mae_24h = round(float(np.mean(np.abs(actuals - p24))), 4)
        return {
            "status": "ok",
            "is_historical_backtest": True,
            "station_id": station_id,
            "hours_evaluated": len(records),
            "actual_total": int(np.sum(actuals)),
            "predicted_1h_total": round(float(np.sum(p1)), 2),
            "predicted_24h_total": round(float(np.sum(p24)), 2),
            "window_mae_1h": mae_1h,
            "window_mae_24h": mae_24h,
            "records": records,
        }

    def predict_all_stations_1h(self, persist: bool = False) -> dict:
        stations = data_service.get_stations_list()
        valid_rows = []
        valid_metas = []
        insufficient = []

        for s in stations:
            st = s["station_id"]
            row, err = self._build_next_1h_feature_row(st)
            if err is not None:
                insufficient.append(err)
            else:
                valid_rows.append(row)
                valid_metas.append(data_service.get_station_detail(st))

        bundle = model_service.get_bundle("1h")
        results = []
        if valid_rows:
            df_batch = pd.DataFrame(valid_rows)
            preds = _score_with_loaded_bundle(df_batch, horizon="1h")
            for row, meta, pred_val in zip(valid_rows, valid_metas, preds):
                p_f = float(pred_val)
                results.append(
                    {
                        "status": "ok",
                        "station_id": meta["station_id"],
                        "short_name": meta["short_name"],
                        "horizon": "1-hour",
                        "model_name": bundle["model_name"],
                        "forecast_origin_pt": meta["last_hour_pt"],
                        "target_timestamp_pt": row["hour_timestamp_pt"],
                        "target_hour": row["hour"],
                        "predicted_sessions": round(p_f, 4),
                        "predicted_sessions_rounded": int(round(p_f)),
                        "historical_hour_avg": round(float(meta["hourly_mean_sessions"][row["hour"]]), 4),
                        "overall_hourly_mean": meta["overall_hourly_mean"],
                        "is_cold_start": meta["is_cold_start"],
                        "cold_start_note": meta["cold_start_note"],
                        "prior_plugged_in_vehicles": row["prior_plugged_in_vehicles_at_hour_start"],
                    }
                )

        results.sort(key=lambda x: x["predicted_sessions"], reverse=True)
        if persist and results:
            log_predictions(results)

        return {
            "horizon": "1-hour",
            "model_name": bundle["model_name"],
            "metric_name": "predicted_session_arrivals",
            "metric_note": "Each station is forecasted for the next clock hour immediately following its own latest observed timestamp (T_last + 1h), using strictly causal features <= T_last.",
            "forecasted_stations_count": len(results),
            "insufficient_history_count": len(insufficient),
            "total_predicted_sessions": round(sum(r["predicted_sessions"] for r in results), 4),
            "predictions": results,
            "insufficient_history_stations": insufficient,
        }

    def predict_all_stations_24h(self, persist: bool = False) -> dict:
        stations = data_service.get_stations_list()
        bundle = model_service.get_bundle("24h")
        valid_frames = []
        valid_metas = []
        insufficient = []

        for s in stations:
            st = s["station_id"]
            df_24, err = self._build_next_24h_feature_rows(st)
            if err is not None:
                insufficient.append(err)
            else:
                valid_frames.append(df_24)
                valid_metas.append(data_service.get_station_detail(st))

        station_summaries = []
        agg_by_step = [0.0] * 24

        if valid_frames:
            df_all = pd.concat(valid_frames, ignore_index=True)
            preds_all = _score_with_loaded_bundle(df_all, horizon="24h")

            offset = 0
            for df_st, meta in zip(valid_frames, valid_metas):
                preds_st = preds_all[offset : offset + 24]
                offset += 24
                hourly_items = []
                for idx, pv in enumerate(preds_st):
                    p_f = float(pv)
                    agg_by_step[idx] += p_f
                    r = df_st.iloc[idx]
                    hr = int(r["hour"])
                    hourly_items.append(
                        {
                            "step_ahead": idx + 1,
                            "target_timestamp_pt": str(r["hour_timestamp_pt"]),
                            "hour": hr,
                            "label": f"{hr:02d}:00",
                            "predicted_sessions": round(p_f, 4),
                            "predicted_sessions_rounded": int(round(p_f)),
                        }
                    )
                tot_24 = round(float(np.sum(preds_st)), 4)
                pk = max(hourly_items, key=lambda x: x["predicted_sessions"])
                station_summaries.append(
                    {
                        "status": "ok",
                        "station_id": meta["station_id"],
                        "short_name": meta["short_name"],
                        "horizon": "24-hour",
                        "model_name": bundle["model_name"],
                        "forecast_origin_pt": meta["last_hour_pt"],
                        "forecast_window_start_pt": hourly_items[0]["target_timestamp_pt"],
                        "forecast_window_end_pt": hourly_items[-1]["target_timestamp_pt"],
                        "total_predicted_sessions_24h": tot_24,
                        "total_predicted_sessions_24h_rounded": int(round(tot_24)),
                        "mean_hourly_predicted_sessions": round(tot_24 / 24.0, 4),
                        "peak_predicted_hour_pt": pk["target_timestamp_pt"],
                        "peak_predicted_hour": pk["hour"],
                        "peak_predicted_sessions": pk["predicted_sessions"],
                        "is_cold_start": meta["is_cold_start"],
                        "cold_start_note": meta["cold_start_note"],
                    }
                )

        station_summaries.sort(key=lambda x: x["total_predicted_sessions_24h"], reverse=True)
        if persist and station_summaries:
            log_predictions(
                [
                    {
                        "station_id": r["station_id"],
                        "horizon": "24-hour",
                        "forecast_origin_pt": r["forecast_origin_pt"],
                        "target_timestamp_pt": r["peak_predicted_hour_pt"],
                        "predicted_sessions": r["total_predicted_sessions_24h"],
                        "predicted_sessions_rounded": r["total_predicted_sessions_24h_rounded"],
                        "model_name": r["model_name"],
                        "is_cold_start": r["is_cold_start"],
                    }
                    for r in station_summaries
                ]
            )

        return {
            "horizon": "24-hour",
            "model_name": bundle["model_name"],
            "metric_name": "predicted_session_arrivals",
            "metric_note": "Strictly <= t-24h day-ahead expected session arrivals for each station's next 24 clock hours (T_last + 1h to T_last + 24h).",
            "forecasted_stations_count": len(station_summaries),
            "insufficient_history_count": len(insufficient),
            "total_predicted_sessions_24h_network": round(sum(r["total_predicted_sessions_24h"] for r in station_summaries), 2),
            "network_hourly_step_totals": [
                {"step_ahead": i + 1, "total_predicted_sessions": round(agg_by_step[i], 3)}
                for i in range(24)
            ],
            "predictions": station_summaries,
            "insufficient_history_stations": insufficient,
        }


prediction_service = PredictionService()
