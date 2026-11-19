import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    CACHE_DIR,
    CLEANED_CSV,
    HOURLY_FEATURES_CSV,
    MIN_HISTORY_HOURS_FOR_PREDICTION,
)
from backend.services.model_service import model_service

CACHE_FILE = os.path.join(CACHE_DIR, "dashboard_data_cache.joblib")
CACHE_SCHEMA_VERSION = 1


def _source_fingerprint() -> dict:
    """File size + mtime fingerprint to ensure derived cache always reflects the read-only source CSVs."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cleaned_size": os.path.getsize(CLEANED_CSV),
        "cleaned_mtime": int(os.path.getmtime(CLEANED_CSV)),
        "features_size": os.path.getsize(HOURLY_FEATURES_CSV),
        "features_mtime": int(os.path.getmtime(HOURLY_FEATURES_CSV)),
    }


class DataService:
    """
    Read-only historical analytics & recent-window service.
    Builds a compact derived cache once from Phase 3 & Phase 4 CSVs so API calls
    never reload the 427 MB hourly feature CSV repeatedly.
    Never modifies any upstream dataset.
    """

    def __init__(self):
        self._cache: Optional[dict] = None

    def initialize(self) -> None:
        if self._cache is not None:
            return
        os.makedirs(CACHE_DIR, exist_ok=True)
        fp = _source_fingerprint()

        if os.path.exists(CACHE_FILE):
            try:
                loaded = joblib.load(CACHE_FILE)
                if loaded.get("fingerprint") == fp:
                    self._cache = loaded
                    return
            except Exception:
                pass

        self._cache = self._build_cache(fp)
        joblib.dump(self._cache, CACHE_FILE, compress=3)

    def _build_cache(self, fingerprint: dict) -> dict:
        if not os.path.exists(CLEANED_CSV):
            raise FileNotFoundError(f"Cleaned dataset missing: {CLEANED_CSV}")
        if not os.path.exists(HOURLY_FEATURES_CSV):
            raise FileNotFoundError(f"Hourly feature dataset missing: {HOURLY_FEATURES_CSV}")

        seen_stations = model_service.get_seen_stations()

        # ------------------------------------------------------------------
        # 1. Read compact subset of Phase 3 cleaned CSV (session-level facts)
        # ------------------------------------------------------------------
        clean_cols = [
            "station_id",
            "Address 1",
            "Latitude",
            "Longitude",
            "port_type_clean",
            "start_datetime_pt",
            "start_date",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "year",
            "month",
            "energy_kwh",
            "charging_duration_min",
            "total_duration_min",
            "flag_duration_unreliable",
            "flag_occupancy_interval_unreliable",
            "occupancy_start_utc",
            "occupancy_end_utc",
            "active_charging_end_utc",
        ]
        dfc = pd.read_csv(CLEANED_CSV, usecols=clean_cols, low_memory=False)
        total_sessions = int(len(dfc))
        total_energy_kwh = round(float(dfc["energy_kwh"].sum()), 2)
        station_count = int(dfc["station_id"].nunique())

        # Hourly historical distribution (0..23)
        n_active_days = int(dfc["start_date"].nunique())
        n_active_wkdays = int(dfc.loc[~dfc["is_weekend"].astype(bool), "start_date"].nunique())
        n_active_wkends = int(dfc.loc[dfc["is_weekend"].astype(bool), "start_date"].nunique())

        hr_grp = dfc.groupby("hour_of_day").agg(
            sessions=("energy_kwh", "size"),
            total_energy_kwh=("energy_kwh", "sum"),
            mean_energy_kwh=("energy_kwh", "mean"),
        ).reindex(range(24), fill_value=0)
        wkday_hr = (
            dfc.loc[~dfc["is_weekend"].astype(bool)]
            .groupby("hour_of_day")
            .size()
            .reindex(range(24), fill_value=0)
            / max(n_active_wkdays, 1)
        )
        wkend_hr = (
            dfc.loc[dfc["is_weekend"].astype(bool)]
            .groupby("hour_of_day")
            .size()
            .reindex(range(24), fill_value=0)
            / max(n_active_wkends, 1)
        )

        hourly_history = [
            {
                "hour": int(h),
                "label": f"{h:02d}:00",
                "sessions": int(hr_grp.loc[h, "sessions"]),
                "pct_share": round(float(hr_grp.loc[h, "sessions"]) / total_sessions * 100.0, 2),
                "total_energy_kwh": round(float(hr_grp.loc[h, "total_energy_kwh"]), 2),
                "mean_energy_kwh": round(float(hr_grp.loc[h, "mean_energy_kwh"]), 2),
                "weekday_avg_sessions_per_day": round(float(wkday_hr.loc[h]), 3),
                "weekend_avg_sessions_per_day": round(float(wkend_hr.loc[h]), 3),
            }
            for h in range(24)
        ]

        # Daily historical series (full 2011-2020)
        daily_grp = (
            dfc.groupby("start_date", as_index=False)
            .agg(
                sessions=("energy_kwh", "size"),
                total_energy_kwh=("energy_kwh", "sum"),
                active_stations=("station_id", "nunique"),
                is_weekend=("is_weekend", "first"),
            )
            .sort_values("start_date")
        )
        daily_history = [
            {
                "date": str(r.start_date),
                "sessions": int(r.sessions),
                "total_energy_kwh": round(float(r.total_energy_kwh), 2),
                "active_stations": int(r.active_stations),
                "is_weekend": bool(r.is_weekend),
            }
            for r in daily_grp.itertuples(index=False)
        ]

        # Day-of-week historical summary
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_grp = dfc.groupby("day_of_week").agg(
            sessions=("energy_kwh", "size"),
            total_energy_kwh=("energy_kwh", "sum"),
            active_days=("start_date", "nunique"),
        ).reindex(range(7), fill_value=0)
        day_of_week_history = [
            {
                "day_of_week": int(d),
                "name": dow_names[d],
                "sessions": int(dow_grp.loc[d, "sessions"]),
                "pct_share": round(float(dow_grp.loc[d, "sessions"]) / total_sessions * 100.0, 2),
                "avg_sessions_per_day": round(
                    float(dow_grp.loc[d, "sessions"]) / max(int(dow_grp.loc[d, "active_days"]), 1), 2
                ),
                "total_energy_kwh": round(float(dow_grp.loc[d, "total_energy_kwh"]), 2),
                "is_weekend": d >= 5,
            }
            for d in range(7)
        ]

        # Level 1 vs Level 2 comparison (using cleaned unambiguous port_type_clean)
        level_comparison = []
        valid_dur_c = ~dfc["flag_duration_unreliable"].astype(bool)
        for pt in ["Level 2", "Level 1"]:
            sub = dfc[dfc["port_type_clean"] == pt]
            sub_vd = dfc[(dfc["port_type_clean"] == pt) & valid_dur_c]
            level_comparison.append(
                {
                    "port_type": pt,
                    "sessions": int(len(sub)),
                    "pct_sessions": round(len(sub) / total_sessions * 100.0, 2),
                    "total_energy_kwh": round(float(sub["energy_kwh"].sum()), 2),
                    "pct_energy": round(float(sub["energy_kwh"].sum()) / total_energy_kwh * 100.0, 2),
                    "mean_energy_kwh": round(float(sub["energy_kwh"].mean()), 2),
                    "mean_charging_min": round(float(sub_vd["charging_duration_min"].mean()), 1),
                    "mean_plugged_min": round(float(sub_vd["total_duration_min"].mean()), 1),
                }
            )

        # Monthly series
        dfc["year_month"] = dfc["start_date"].str.slice(0, 7)
        monthly_grp = (
            dfc.groupby("year_month", as_index=False)
            .agg(
                sessions=("energy_kwh", "size"),
                total_energy_kwh=("energy_kwh", "sum"),
                active_stations=("station_id", "nunique"),
            )
            .sort_values("year_month")
        )
        monthly_history = [
            {
                "year_month": str(r.year_month),
                "sessions": int(r.sessions),
                "total_energy_kwh": round(float(r.total_energy_kwh), 2),
                "active_stations": int(r.active_stations),
            }
            for r in monthly_grp.itertuples(index=False)
        ]

        # Precompute per-station metadata from cleaned sessions
        station_session_meta: Dict[str, dict] = {}
        for st, g in dfc.groupby("station_id", sort=True):
            g_vd = g[~g["flag_duration_unreliable"].astype(bool)]
            st_hr = g.groupby("hour_of_day").size().reindex(range(24), fill_value=0)
            st_daily = (
                g.groupby("start_date", as_index=False)
                .agg(sessions=("energy_kwh", "size"), energy_kwh=("energy_kwh", "sum"))
                .sort_values("start_date")
            )
            # Keep last 180 active dates per station for responsive station trend chart
            st_daily_tail = st_daily.tail(180)
            addr_mode = g["Address 1"].dropna().mode()
            station_session_meta[st] = {
                "station_id": st,
                "short_name": st.replace("PALO ALTO CA / ", ""),
                "address": str(addr_mode.iloc[0]) if len(addr_mode) > 0 else "Unknown",
                "latitude": round(float(g["Latitude"].median()), 6),
                "longitude": round(float(g["Longitude"].median()), 6),
                "port_types": sorted(g["port_type_clean"].dropna().unique().tolist()),
                "total_sessions": int(len(g)),
                "total_energy_kwh": round(float(g["energy_kwh"].sum()), 2),
                "mean_energy_kwh": round(float(g["energy_kwh"].mean()), 2),
                "mean_charging_min": round(float(g_vd["charging_duration_min"].mean()), 1) if len(g_vd) else None,
                "mean_plugged_min": round(float(g_vd["total_duration_min"].mean()), 1) if len(g_vd) else None,
                "active_days": int(g["start_date"].nunique()),
                "sessions_per_active_day": round(len(g) / max(int(g["start_date"].nunique()), 1), 2),
                "hourly_total_sessions": [int(st_hr.loc[h]) for h in range(24)],
                "daily_recent": [
                    {
                        "date": str(r.start_date),
                        "sessions": int(r.sessions),
                        "energy_kwh": round(float(r.energy_kwh), 2),
                    }
                    for r in st_daily_tail.itertuples(index=False)
                ],
                "in_training_data": st in seen_stations,
                "is_cold_start": st not in seen_stations,
            }

        # Precompute occupancy intervals per station (valid intervals only) for exact next-hour pre-hour concurrency
        valid_occ = dfc[~dfc["flag_occupancy_interval_unreliable"].astype(bool)].copy()
        valid_occ["occ_s"] = pd.to_datetime(valid_occ["occupancy_start_utc"])
        valid_occ["occ_e"] = pd.to_datetime(valid_occ["occupancy_end_utc"])
        valid_occ["act_e"] = pd.to_datetime(valid_occ["active_charging_end_utc"])
        valid_occ["dur_ok"] = ~valid_occ["flag_duration_unreliable"].astype(bool)

        # ------------------------------------------------------------------
        # 2. Read Phase 4 hourly feature CSV once to extract:
        #    - Exact network-wide station-hour average
        #    - Per-station hourly means (0..23) & overall span
        #    - Per-station recent 504-hour (21-day) causal feature windows
        # ------------------------------------------------------------------
        dff = pd.read_csv(HOURLY_FEATURES_CSV, low_memory=False)
        for col in dff.select_dtypes(include=["float64"]).columns:
            dff[col] = dff[col].astype("float32")
        dff["sessions_count"] = dff["sessions_count"].astype("int32")

        total_station_hours = int(len(dff))
        unique_network_hours = int(dff["hour_timestamp_utc"].nunique())
        avg_sessions_per_station_hour = round(total_sessions / max(total_station_hours, 1), 4)
        avg_sessions_per_network_hour = round(total_sessions / max(unique_network_hours, 1), 2)

        station_recent_panels: Dict[str, pd.DataFrame] = {}
        stations_meta: Dict[str, dict] = {}

        # Sort by station_id + hour_timestamp_utc to slice exact tail windows
        dff_sorted = dff.sort_values(["station_id", "hour_timestamp_utc"])
        for st, g in dff_sorted.groupby("station_id", sort=True):
            meta = dict(station_session_meta[st])
            span_hours = int(len(g))
            first_pt = str(g["hour_timestamp_pt"].iloc[0])
            last_pt = str(g["hour_timestamp_pt"].iloc[-1])
            last_utc_str = str(g["hour_timestamp_utc"].iloc[-1])
            last_utc_ts = pd.Timestamp(last_utc_str)
            next_utc_ts = last_utc_ts + pd.Timedelta(hours=1)

            # Exact pre-hour occupancy at T_last + 1h from cleaned sessions (strictly start < next_utc < end)
            st_occ = valid_occ[valid_occ["station_id"] == st]
            prior_plugged_next = int(((st_occ["occ_s"] < next_utc_ts) & (st_occ["occ_e"] > next_utc_ts)).sum())
            prior_active_next = int(
                ((st_occ["occ_s"] < next_utc_ts) & (st_occ["act_e"] > next_utc_ts) & st_occ["dur_ok"]).sum()
            )

            # Station hourly mean arrivals across all observed hours in its span
            hr_means = g.groupby("hour")["sessions_count"].mean().reindex(range(24), fill_value=0.0)
            hourly_mean_list = [round(float(hr_means.loc[h]), 4) for h in range(24)]
            top3_hrs = sorted(range(24), key=lambda h: hr_means.loc[h], reverse=True)[:3]

            # Causal counts needed at T_last + 1h
            total_energy_st = float(g["energy_kwh_arrival_sum"].sum())
            overall_hr_mean = float(g["sessions_count"].mean())
            recent_168_mean = float(g["sessions_count"].tail(168).mean())
            recent_24_sum = int(g["sessions_count"].tail(24).sum())

            meta.update(
                {
                    "first_hour_pt": first_pt,
                    "last_hour_pt": last_pt,
                    "last_hour_utc": last_utc_str,
                    "span_hours": span_hours,
                    "has_sufficient_history": span_hours >= MIN_HISTORY_HOURS_FOR_PREDICTION,
                    "overall_hourly_mean": round(overall_hr_mean, 4),
                    "recent_168h_hourly_mean": round(recent_168_mean, 4),
                    "recent_24h_sessions": recent_24_sum,
                    "hourly_mean_sessions": hourly_mean_list,
                    "peak_hours": [int(h) for h in top3_hrs],
                    "peak_hour_mean": round(float(hr_means.loc[top3_hrs[0]]), 4),
                    "prior_plugged_at_next_hour": prior_plugged_next,
                    "prior_active_at_next_hour": prior_active_next,
                    "total_energy_from_panel": round(total_energy_st, 2),
                }
            )
            if meta["is_cold_start"]:
                if not meta["has_sufficient_history"]:
                    meta["cold_start_note"] = (
                        f"Cold-start station (not present in 2011–2018 training data) with only {span_hours} "
                        f"recorded hour(s). Minimum {MIN_HISTORY_HOURS_FOR_PREDICTION} hours required for lag-based forecasts."
                    )
                else:
                    meta["cold_start_note"] = (
                        f"Cold-start station (opened in Dec 2020; not present in 2011–2018 training data). "
                        f"Forecasts rely on {span_hours} observed hours and generalized temporal patterns; interpret with caution."
                    )
            else:
                meta["cold_start_note"] = None

            stations_meta[st] = meta
            # Keep last 504 hours (21 days) per station — enough for lag_336h + rolling_168h + 48h backtest!
            station_recent_panels[st] = g.tail(504).reset_index(drop=True)

        stations_ranked = sorted(
            stations_meta.values(), key=lambda x: x["total_sessions"], reverse=True
        )

        summary = {
            "total_sessions": total_sessions,
            "total_energy_kwh": total_energy_kwh,
            "total_energy_mwh": round(total_energy_kwh / 1000.0, 2),
            "station_count": station_count,
            "historical_training_stations_count": sum(1 for s in stations_meta.values() if s["in_training_data"]),
            "cold_start_stations_count": sum(1 for s in stations_meta.values() if s["is_cold_start"]),
            "total_station_hours": total_station_hours,
            "unique_network_hours": unique_network_hours,
            "avg_sessions_per_station_hour": avg_sessions_per_station_hour,
            "avg_sessions_per_network_hour": avg_sessions_per_network_hour,
            "date_range_pt": {
                "start": str(dff["hour_timestamp_pt"].iloc[0]),
                "end": str(dff["hour_timestamp_pt"].iloc[-1]),
            },
            "level_comparison": level_comparison,
            "top_10_stations": [
                {
                    "station_id": s["station_id"],
                    "short_name": s["short_name"],
                    "total_sessions": s["total_sessions"],
                    "total_energy_kwh": s["total_energy_kwh"],
                    "sessions_per_active_day": s["sessions_per_active_day"],
                    "mean_energy_kwh": s["mean_energy_kwh"],
                }
                for s in stations_ranked[:10]
            ],
        }

        return {
            "fingerprint": fingerprint,
            "summary": summary,
            "hourly_history": hourly_history,
            "daily_history": daily_history,
            "day_of_week_history": day_of_week_history,
            "monthly_history": monthly_history,
            "stations_meta": stations_meta,
            "station_recent_panels": station_recent_panels,
        }

    def get_summary(self) -> dict:
        self.initialize()
        return self._cache["summary"]

    def get_hourly_history(self) -> List[dict]:
        self.initialize()
        return self._cache["hourly_history"]

    def get_daily_history(self, year: Optional[int] = None, limit: Optional[int] = None) -> List[dict]:
        self.initialize()
        rows = self._cache["daily_history"]
        if year is not None:
            prefix = f"{int(year)}-"
            rows = [r for r in rows if r["date"].startswith(prefix)]
        if limit is not None and limit > 0:
            rows = rows[-int(limit):]
        return rows

    def get_day_of_week_history(self) -> List[dict]:
        self.initialize()
        return self._cache["day_of_week_history"]

    def get_monthly_history(self) -> List[dict]:
        self.initialize()
        return self._cache["monthly_history"]

    def get_stations_list(self) -> List[dict]:
        self.initialize()
        stations = sorted(
            self._cache["stations_meta"].values(),
            key=lambda x: x["total_sessions"],
            reverse=True,
        )
        # Return compact listing without bulky daily arrays
        return [
            {
                "station_id": s["station_id"],
                "short_name": s["short_name"],
                "address": s["address"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "port_types": s["port_types"],
                "total_sessions": s["total_sessions"],
                "total_energy_kwh": s["total_energy_kwh"],
                "mean_energy_kwh": s["mean_energy_kwh"],
                "active_days": s["active_days"],
                "sessions_per_active_day": s["sessions_per_active_day"],
                "first_hour_pt": s["first_hour_pt"],
                "last_hour_pt": s["last_hour_pt"],
                "span_hours": s["span_hours"],
                "in_training_data": s["in_training_data"],
                "is_cold_start": s["is_cold_start"],
                "has_sufficient_history": s["has_sufficient_history"],
                "cold_start_note": s["cold_start_note"],
                "overall_hourly_mean": s["overall_hourly_mean"],
                "peak_hours": s["peak_hours"],
            }
            for s in stations
        ]

    def get_station_detail(self, station_id: str) -> Optional[dict]:
        self.initialize()
        return self._cache["stations_meta"].get(station_id)

    def get_station_recent_panel(self, station_id: str) -> Optional[pd.DataFrame]:
        self.initialize()
        panel = self._cache["station_recent_panels"].get(station_id)
        return panel.copy() if panel is not None else None


data_service = DataService()
