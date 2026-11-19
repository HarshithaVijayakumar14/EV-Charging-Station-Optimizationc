import hashlib
import os
from datetime import date, timedelta
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def get_us_federal_holidays(years) -> Set[date]:
    """Deterministic US federal holiday dates (observed) for given years (pure calendar rules)."""
    holidays = set()

    def nth_weekday(year, month, weekday, n):
        d1 = date(year, month, 1)
        offset = (weekday - d1.weekday()) % 7
        return d1 + timedelta(days=offset + 7 * (n - 1))

    def last_weekday(year, month, weekday):
        if month == 12:
            next_m = date(year + 1, 1, 1)
        else:
            next_m = date(year, month + 1, 1)
        last_d = next_m - timedelta(days=1)
        offset = (last_d.weekday() - weekday) % 7
        return last_d - timedelta(days=offset)

    def observed(d):
        if d.weekday() == 5:  # Saturday -> Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:  # Sunday -> Monday
            return d + timedelta(days=1)
        return d

    for y in years:
        for m, day_num in [(1, 1), (7, 4), (11, 11), (12, 25)]:
            actual = date(y, m, day_num)
            holidays.add(actual)
            holidays.add(observed(actual))
        holidays.add(nth_weekday(y, 1, 0, 3))   # MLK Jr. Day
        holidays.add(nth_weekday(y, 2, 0, 3))   # Presidents' Day
        holidays.add(last_weekday(y, 5, 0))     # Memorial Day
        holidays.add(nth_weekday(y, 9, 0, 1))   # Labor Day
        holidays.add(nth_weekday(y, 10, 0, 2))  # Columbus Day
        holidays.add(nth_weekday(y, 11, 3, 4))  # Thanksgiving
    return holidays

WORKSPACE = r"c:\Users\harsh\OneDrive\Desktop\ML"
RAW_CSV = os.path.join(WORKSPACE, "EVChargingStationUsage.csv")
CLEANED_CSV = os.path.join(WORKSPACE, "dataset", "processed", "ev_charging_cleaned.csv")
FEATURES_CSV = os.path.join(WORKSPACE, "dataset", "processed", "ev_charging_hourly_features.csv")
MODELS_DIR = os.path.join(WORKSPACE, "models")
EVAL_DIR = os.path.join(WORKSPACE, "reports", "model_evaluation")

TARGET_COL = "sessions_count"

# 1-hour-ahead causal feature list (all strictly <= t-1h or real-time pre-hour occupancy at t:00:00)
# Raw `year` is deliberately excluded so tree models do not split on non-repeating calendar years;
# policy regime (`is_paid_era`), `station_age_days`, and causal rolling means capture level shifts.
FEATURES_1H: List[str] = [
    "station_encoded",
    "hour",
    "day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_us_federal_holiday",
    "is_working_day",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_paid_era",
    "station_age_days",
    "station_cum_sessions_prior",
    "station_expanding_mean_sessions",
    "station_hour_expanding_mean_sessions",
    "hours_since_last_session",
    "station_hist_mean_energy_per_session",
    "station_hist_mean_plugged_dur_min",
    "station_hist_mean_active_chg_min",
    "prior_plugged_in_vehicles_at_hour_start",
    "prior_active_charging_vehicles_at_hour_start",
    "lag_1h_sessions",
    "lag_2h_sessions",
    "lag_3h_sessions",
    "lag_24h_sessions",
    "lag_168h_sessions",
    "rolling_mean_24h_sessions",
    "rolling_sum_24h_sessions",
    "rolling_mean_168h_sessions",
    "rolling_sum_168h_sessions",
    "lag_1h_energy_kwh",
    "lag_24h_energy_kwh",
    "lag_168h_energy_kwh",
    "rolling_mean_24h_energy_kwh",
    "rolling_mean_168h_energy_kwh",
]

# 24-hour-ahead strictly causal feature list (ONLY info available at <= t - 24h + deterministic calendar for hour t)
FEATURES_24H: List[str] = [
    "station_encoded",
    "hour",
    "day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_us_federal_holiday",
    "is_working_day",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_paid_era",
    "station_age_days",
    "station_cum_sessions_at_t24",
    "station_expanding_mean_at_t24_sessions",
    "station_hour_expanding_mean_sessions",  # same hour on strictly prior days (<= t - 24h)
    "lag_24h_sessions",
    "lag_48h_sessions",
    "lag_168h_sessions",
    "lag_336h_sessions",
    "rolling_mean_24h_at_t24_sessions",
    "rolling_sum_24h_at_t24_sessions",
    "rolling_mean_168h_at_t24_sessions",
    "rolling_sum_168h_at_t24_sessions",
    "lag_24h_energy_kwh",
    "lag_168h_energy_kwh",
    "rolling_mean_168h_at_t24_energy_kwh",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hourly_features(path: str = FEATURES_CSV) -> pd.DataFrame:
    """Load Phase 4 hourly feature dataset cleanly in chronological order with compact dtypes."""
    df = pd.read_csv(path, low_memory=False)
    # Downcast floats/ints to float32/int32 to keep memory footprint lean for RF/XGB on 1.53M rows
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in ["hour", "day", "day_of_week", "month", "is_weekend", "is_us_federal_holiday", "is_working_day", "is_paid_era"]:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    for col in ["year", "prior_plugged_in_vehicles_at_hour_start", "prior_active_charging_vehicles_at_hour_start"]:
        if col in df.columns:
            df[col] = df[col].astype("int16")
    df["sessions_count"] = df["sessions_count"].astype("int32")
    # Ensure exact chronological sort (hour_timestamp_utc, station_id)
    df = df.sort_values(["hour_timestamp_utc", "station_id"]).reset_index(drop=True)
    return df


def add_24h_ahead_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive strictly <= t - 24h causal features per station on the contiguous UTC hourly spine.
    Never modifies the input CSV on disk; returns DataFrame sorted chronologically by (hour_timestamp_utc, station_id).
    """
    work = df.sort_values(["station_id", "hour_timestamp_utc"]).reset_index(drop=True)
    g = work.groupby("station_id", sort=False)

    s_shift24 = g["sessions_count"].shift(24)
    e_shift24 = g["energy_kwh_arrival_sum"].shift(24)

    work["lag_48h_sessions"] = g["sessions_count"].shift(48).astype("float32")
    work["lag_336h_sessions"] = g["sessions_count"].shift(336).astype("float32")

    work["_s24"] = s_shift24.astype("float32")
    work["_e24"] = e_shift24.astype("float32")
    gs = work.groupby("station_id", sort=False)

    work["rolling_mean_24h_at_t24_sessions"] = gs["_s24"].transform(
        lambda x: x.rolling(window=24, min_periods=24).mean()
    ).astype("float32")
    work["rolling_sum_24h_at_t24_sessions"] = gs["_s24"].transform(
        lambda x: x.rolling(window=24, min_periods=24).sum()
    ).astype("float32")
    work["rolling_mean_168h_at_t24_sessions"] = gs["_s24"].transform(
        lambda x: x.rolling(window=168, min_periods=168).mean()
    ).astype("float32")
    work["rolling_sum_168h_at_t24_sessions"] = gs["_s24"].transform(
        lambda x: x.rolling(window=168, min_periods=168).sum()
    ).astype("float32")
    work["rolling_mean_168h_at_t24_energy_kwh"] = gs["_e24"].transform(
        lambda x: x.rolling(window=168, min_periods=168).mean()
    ).astype("float32")

    work["station_cum_sessions_at_t24"] = gs["_s24"].cumsum().fillna(0.0).astype("float32")
    work["station_expanding_mean_at_t24_sessions"] = gs["_s24"].transform(
        lambda x: x.expanding(min_periods=1).mean()
    ).astype("float32")

    work.drop(columns=["_s24", "_e24"], inplace=True)
    work = work.sort_values(["hour_timestamp_utc", "station_id"]).reset_index(drop=True)
    return work


def verify_24h_no_future_leakage(df: pd.DataFrame) -> bool:
    """
    Adversarial perturbation proof for 24-hour-ahead features:
    Mutate targets across [t* - 23 .. end] by +9999 on a busy station and verify
    that 100% of 24h-ahead features at t* remain bit-for-bit identical.
    """
    sample_st = "PALO ALTO CA / HAMILTON #2"
    sub = df[df["station_id"] == sample_st].sort_values("hour_timestamp_utc").reset_index(drop=True)
    sub = sub.iloc[:2000].copy()
    t_star = 1000

    orig = add_24h_ahead_features(sub)
    mut_in = sub.copy()
    # Mutate all hours within the 24-hour forbidden window [t* - 23 .. end]
    mut_in.loc[t_star - 23 :, "sessions_count"] += 9999
    mut_in.loc[t_star - 23 :, "energy_kwh_arrival_sum"] += 9999.0
    # Also verify station_hour_expanding_mean_sessions recomputed cleanly on mutated input
    mut_in["station_hour_expanding_mean_sessions"] = (
        mut_in.groupby(["station_id", "hour"], sort=False)["sessions_count"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .astype("float32")
    )
    mut_in["lag_24h_sessions"] = mut_in["sessions_count"].shift(24).astype("float32")
    mut_in["lag_168h_sessions"] = mut_in["sessions_count"].shift(168).astype("float32")
    mut_in["lag_24h_energy_kwh"] = mut_in["energy_kwh_arrival_sum"].shift(24).astype("float32")
    mut_in["lag_168h_energy_kwh"] = mut_in["energy_kwh_arrival_sum"].shift(168).astype("float32")
    mut = add_24h_ahead_features(mut_in)

    check_cols = [c for c in FEATURES_24H if c != "station_encoded"]
    for c in check_cols:
        v1 = orig.loc[t_star, c]
        v2 = mut.loc[t_star, c]
        if pd.isna(v1) and pd.isna(v2):
            continue
        if v1 != v2:
            raise AssertionError(f"24h-ahead leakage detected in feature '{c}': {v1} vs {v2}")
    return True


class TabularPreprocessor:
    """
    Strict training-only preprocessor:
    - Fits station_id -> integer ordinal encoding on training split only (unseen stations -> -1).
    - Fits per-feature medians on training split only for imputing warm-up NaNs.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.station_to_code: Dict[str, int] = {}
        self.medians: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame) -> "TabularPreprocessor":
        stations = sorted(df_train["station_id"].unique().tolist())
        self.station_to_code = {st: idx for idx, st in enumerate(stations)}
        enc = df_train["station_id"].map(self.station_to_code).fillna(-1).astype("int32")

        for col in self.feature_cols:
            if col == "station_encoded":
                self.medians[col] = -1.0
            else:
                med = float(np.nanmedian(df_train[col].to_numpy(dtype="float64")))
                self.medians[col] = 0.0 if np.isnan(med) else med
        return self

    def transform(self, df: pd.DataFrame, impute: bool = True) -> np.ndarray:
        n = len(df)
        X = np.empty((n, len(self.feature_cols)), dtype=np.float32)
        for j, col in enumerate(self.feature_cols):
            if col == "station_encoded":
                arr = df["station_id"].map(self.station_to_code).fillna(-1).to_numpy(dtype=np.float32)
            else:
                arr = df[col].to_numpy(dtype=np.float32, copy=True)
                if impute:
                    mask = np.isnan(arr)
                    if mask.any():
                        arr[mask] = np.float32(self.medians[col])
            X[:, j] = arr
        return X


class HistoricalMeanBaseline:
    """
    Training-only Historical Mean Baseline:
    Predicts mean sessions_count by (station_id, hour, is_weekend), falling back cleanly to
    (hour, is_weekend) mean for unseen/cold-start stations, and finally global training mean.
    """

    def __init__(self):
        self.global_mean_: float = 0.0
        self.hour_wknd_means_: Dict[Tuple[int, int], float] = {}
        self.st_hour_wknd_means_: Dict[Tuple[str, int, int], float] = {}

    def fit(self, df_train: pd.DataFrame) -> "HistoricalMeanBaseline":
        self.global_mean_ = float(df_train[TARGET_COL].mean())
        hw = df_train.groupby(["hour", "is_weekend"])[TARGET_COL].mean()
        self.hour_wknd_means_ = {(int(k[0]), int(k[1])): float(v) for k, v in hw.items()}
        shw = df_train.groupby(["station_id", "hour", "is_weekend"])[TARGET_COL].mean()
        self.st_hour_wknd_means_ = {(str(k[0]), int(k[1]), int(k[2])): float(v) for k, v in shw.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Vectorized merge lookup
        shw_df = pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in self.st_hour_wknd_means_.items()],
            columns=["station_id", "hour", "is_weekend", "_p_st"],
        )
        hw_df = pd.DataFrame(
            [(k[0], k[1], v) for k, v in self.hour_wknd_means_.items()],
            columns=["hour", "is_weekend", "_p_hw"],
        )
        keys = df[["station_id", "hour", "is_weekend"]].copy()
        m = keys.merge(shw_df, on=["station_id", "hour", "is_weekend"], how="left")
        m = m.merge(hw_df, on=["hour", "is_weekend"], how="left")
        preds = m["_p_st"].fillna(m["_p_hw"]).fillna(self.global_mean_).to_numpy(dtype=np.float64)
        return np.clip(preds, 0.0, None)


class SeasonalNaiveBaseline:
    """
    Seasonal Naive Baseline combining previous 24-hour and 168-hour demand:
    y_hat = w_24 * lag_24h + (1 - w_24) * lag_168h
    When one lag is NaN, uses the available lag; when both are NaN, falls back to training global mean.
    """

    def __init__(self, w_24: float = 0.5):
        self.w_24 = float(w_24)
        self.fallback_mean_: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "SeasonalNaiveBaseline":
        self.fallback_mean_ = float(df_train[TARGET_COL].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        l24 = df["lag_24h_sessions"].to_numpy(dtype=np.float64)
        l168 = df["lag_168h_sessions"].to_numpy(dtype=np.float64)
        both = (~np.isnan(l24)) & (~np.isnan(l168))
        only24 = (~np.isnan(l24)) & np.isnan(l168)
        only168 = np.isnan(l24) & (~np.isnan(l168))

        out = np.full(len(df), self.fallback_mean_, dtype=np.float64)
        out[both] = self.w_24 * l24[both] + (1.0 - self.w_24) * l168[both]
        out[only24] = l24[only24]
        out[only168] = l168[only168]
        return np.clip(out, 0.0, None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    # Guard against constant y_true in tiny slices
    r2 = float(r2_score(y_t, y_p)) if np.var(y_t) > 0 else float("nan")
    actual_sum = float(y_t.sum())
    pred_sum = float(y_p.sum())
    bias_pct = ((pred_sum - actual_sum) / actual_sum * 100.0) if actual_sum > 0 else float("nan")
    return {
        "n": int(len(y_t)),
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 5) if not np.isnan(r2) else None,
        "actual_sum": round(actual_sum, 2),
        "predicted_sum": round(pred_sum, 2),
        "bias_pct": round(bias_pct, 2) if not np.isnan(bias_pct) else None,
    }
