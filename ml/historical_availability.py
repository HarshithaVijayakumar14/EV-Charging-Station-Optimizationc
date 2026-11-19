"""
ml/historical_availability.py
Reconstructs hourly historical station availability from validated session intervals
and the observed charging point inventory.
Ensures:
occupied_point_count + free_observed_point_count + unknown_point_count == observed_point_count
Never claims unobserved capacity or negative availability.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY_FEATURES_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "ev_charging_hourly_features.csv")
INVENTORY_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "charging_point_inventory.csv")
HISTORICAL_AVAILABILITY_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "historical_station_availability.csv")
HISTORICAL_AVAILABILITY_RECENT_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "historical_availability_recent.csv")


def reconstruct_historical_availability(
    hourly_features_csv: str = HOURLY_FEATURES_CSV,
    inventory_csv: str = INVENTORY_CSV,
    output_csv: str = HISTORICAL_AVAILABILITY_CSV,
    start_year: Optional[int] = None,
) -> pd.DataFrame:
    if not os.path.exists(hourly_features_csv):
        raise FileNotFoundError(f"Hourly feature dataset not found: {hourly_features_csv}")
    if not os.path.exists(inventory_csv):
        raise FileNotFoundError(f"Charging point inventory not found: {inventory_csv}")

    inv_df = pd.read_csv(inventory_csv)
    # Active points count per station at end of data (or at least 1)
    active_pts_map = inv_df.groupby("station_id")["is_active_end_of_data"].sum().to_dict()
    for st in inv_df["station_id"].unique():
        if active_pts_map.get(st, 0) == 0:
            active_pts_map[st] = 1

    # Load hourly features
    cols = [
        "hour_timestamp_pt",
        "station_id",
        "prior_plugged_in_vehicles_at_hour_start",
        "sessions_count",
    ]
    df_h = pd.read_csv(hourly_features_csv, usecols=cols, low_memory=False)

    if start_year is not None:
        df_h = df_h[df_h["hour_timestamp_pt"].str.slice(0, 4).astype(int) >= start_year].copy()

    # Calculate fields
    obs_pts = df_h["station_id"].map(active_pts_map).fillna(2).astype(int)
    plugged = df_h["prior_plugged_in_vehicles_at_hour_start"].fillna(0).astype(int)

    # Exceeded observed points indicator
    exceeded = plugged > obs_pts

    # Occupied points is strictly bounded by observed_point_count
    occupied = np.minimum(plugged, obs_pts)

    # Unknown points (e.g. if an unreliable record or inventory ambiguity occurred)
    # For clean data, unknown = 0; if exceeded, the extra vehicle indicates unmodeled inventory
    unknown = np.zeros(len(df_h), dtype=int)

    # Free points: exactly obs_pts - occupied - unknown
    free = np.maximum(0, obs_pts - occupied - unknown)

    # Data quality flags
    flags = np.where(exceeded, "OCCUPANCY_EXCEEDED_OBSERVED_POINTS", "")

    # Confidence: High for established sites with normal load, Medium if exceeded, Low for sparse/cold-start
    confidence = np.where(
        exceeded,
        "Medium",
        np.where(obs_pts <= 1, "Medium", "High"),
    )

    out = pd.DataFrame(
        {
            "timestamp": df_h["hour_timestamp_pt"],
            "station_id": df_h["station_id"],
            "observed_point_count": obs_pts,
            "occupied_point_count": occupied,
            "free_observed_point_count": free,
            "unknown_point_count": unknown,
            "availability_confidence": confidence,
            "data_quality_flags": flags,
        }
    )

    # Mathematical identity verification
    check = out["occupied_point_count"] + out["free_observed_point_count"] + out["unknown_point_count"]
    assert (check == out["observed_point_count"]).all(), "Conservation violation: occupied + free + unknown != observed!"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


class HistoricalAvailabilityService:
    def __init__(self, availability_csv: str = HISTORICAL_AVAILABILITY_CSV):
        self.availability_csv = availability_csv
        self._tail_cache: Optional[Dict[str, pd.DataFrame]] = None
        self._df: Optional[pd.DataFrame] = None

    def initialize(self) -> None:
        if self._tail_cache is not None:
            return

        target_file = self.availability_csv
        if not os.path.exists(target_file):
            if os.path.exists(HISTORICAL_AVAILABILITY_RECENT_CSV):
                target_file = HISTORICAL_AVAILABILITY_RECENT_CSV
            elif os.path.exists(HOURLY_FEATURES_CSV):
                reconstruct_historical_availability(output_csv=self.availability_csv)
            else:
                target_file = HISTORICAL_AVAILABILITY_RECENT_CSV

        # Cache last 168 hours (7 days) per station for fast API queries
        if os.path.exists(target_file):
            df = pd.read_csv(target_file, low_memory=False)
            self._tail_cache = {}
            for st, g in df.groupby("station_id"):
                self._tail_cache[st] = g.tail(168).reset_index(drop=True)
        else:
            self._tail_cache = {}

    def _ensure_full_df(self) -> pd.DataFrame:
        if self._df is None:
            if os.path.exists(self.availability_csv):
                self._df = pd.read_csv(self.availability_csv, low_memory=False)
            elif os.path.exists(HISTORICAL_AVAILABILITY_RECENT_CSV):
                self._df = pd.read_csv(HISTORICAL_AVAILABILITY_RECENT_CSV, low_memory=False)
            else:
                self._df = pd.DataFrame(columns=[
                    "timestamp", "station_id", "observed_point_count",
                    "occupied_point_count", "free_observed_point_count",
                    "unknown_point_count", "availability_confidence", "data_quality_flags"
                ])
        return self._df

    def get_recent_station_availability(self, station_id: str, limit: int = 48) -> List[Dict]:
        self.initialize()
        tail = self._tail_cache.get(station_id)
        if tail is None or len(tail) == 0:
            return []
        sub = tail.tail(min(limit, len(tail)))
        return sub.to_dict(orient="records")

    def query_availability(
        self,
        station_id: Optional[str] = None,
        date: Optional[str] = None,
        hour: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict]:
        self.initialize()
        # Fast path: station_id only, no specific historical date -> use tail cache
        if station_id and not date and hour is None:
            return self.get_recent_station_availability(station_id, limit=limit)

        # If specific date or cross-station query requested:
        df = self._ensure_full_df()
        mask = pd.Series(True, index=df.index)
        if station_id:
            mask &= (df["station_id"] == station_id)
        if date:
            mask &= df["timestamp"].str.startswith(date)
        if hour is not None:
            hr_str = f"{int(hour):02d}"
            mask &= (df["timestamp"].str.slice(11, 13) == hr_str)

        filtered = df[mask]
        if len(filtered) > limit:
            filtered = filtered.tail(limit)
        return filtered.to_dict(orient="records")


historical_availability_service = HistoricalAvailabilityService()

if __name__ == "__main__":
    print("Reconstructing historical station availability...")
    df_avail = reconstruct_historical_availability()
    print(f"Generated {len(df_avail):,} hourly availability records.")
    print("Sample records:")
    print(df_avail.head(10).to_string())
    print("\nSummary statistics:")
    print(df_avail[["observed_point_count", "occupied_point_count", "free_observed_point_count"]].describe())
