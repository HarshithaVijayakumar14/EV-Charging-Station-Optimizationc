import hashlib
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

WORKSPACE = r"c:\Users\harsh\OneDrive\Desktop\ML"
RAW_ROOT_CSV = os.path.join(WORKSPACE, "EVChargingStationUsage.csv")
CLEANED_CSV = os.path.join(WORKSPACE, "dataset", "processed", "ev_charging_cleaned.csv")
OUTPUT_CSV = os.path.join(WORKSPACE, "dataset", "processed", "ev_charging_hourly_features.csv")
SUMMARY_JSON = r"C:\Users\harsh\.gemini\antigravity\brain\7aee0d63-4143-475b-8d08-cc48808ea73c\scratch\phase4_feature_summary.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_us_federal_holidays(years) -> set:
    """Deterministic US federal holiday dates (observed) for given years (pure calendar rules)."""
    holidays = set()

    def nth_weekday(year, month, weekday, n):
        # weekday: 0=Mon .. 6=Sun
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
        # Fixed holidays (both actual and observed if weekend)
        for m, day_num in [(1, 1), (7, 4), (11, 11), (12, 25)]:
            actual = date(y, m, day_num)
            holidays.add(actual)
            holidays.add(observed(actual))
        # Floating Monday / Thursday holidays
        holidays.add(nth_weekday(y, 1, 0, 3))   # MLK Jr. Day (3rd Mon Jan)
        holidays.add(nth_weekday(y, 2, 0, 3))   # Presidents' Day (3rd Mon Feb)
        holidays.add(last_weekday(y, 5, 0))     # Memorial Day (last Mon May)
        holidays.add(nth_weekday(y, 9, 0, 1))   # Labor Day (1st Mon Sep)
        holidays.add(nth_weekday(y, 10, 0, 2))  # Columbus Day (2nd Mon Oct)
        holidays.add(nth_weekday(y, 11, 3, 4))  # Thanksgiving (4th Thu Nov)
    return holidays


def utc_to_pacific_with_offsets(utc_ser: pd.Series):
    """Return (pacific_local_naive_ts, offset_hours_series) for US Pacific DST rules."""
    years = utc_ser.dt.year.unique()
    offsets_hr = pd.Series(-8, index=utc_ser.index, dtype="int8")
    for y in years:
        y_int = int(y)
        mar1 = datetime(y_int, 3, 1)
        first_sun_mar = 1 + (6 - mar1.weekday()) % 7
        dst_start_utc = pd.Timestamp(datetime(y_int, 3, first_sun_mar + 7, 10, 0))
        nov1 = datetime(y_int, 11, 1)
        first_sun_nov = 1 + (6 - nov1.weekday()) % 7
        dst_end_utc = pd.Timestamp(datetime(y_int, 11, first_sun_nov, 9, 0))
        in_dst = (utc_ser >= dst_start_utc) & (utc_ser < dst_end_utc)
        offsets_hr.loc[in_dst] = -7
    pt_naive = utc_ser + pd.to_timedelta(offsets_hr.astype("int64"), unit="h")
    return pt_naive, offsets_hr


def compute_prior_occupancy_at_hour_starts(
    spine_utc_ns: np.ndarray,
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
) -> np.ndarray:
    """
    Count sessions with start < t < end for every hour-start timestamp t in spine_utc_ns.
    Because strict inequality start < t is enforced, ONLY sessions arriving strictly BEFORE
    hour t (already plugged in before t:00:00) are counted — zero leakage from arrivals in hour t.
    Uses O((N + M) log N) binary search via searchsorted.
    """
    if len(starts_ns) == 0:
        return np.zeros(len(spine_utc_ns), dtype="int16")
    s_sorted = np.sort(starts_ns)
    e_sorted = np.sort(ends_ns)
    # Count(start < t) = searchsorted(s_sorted, t, side='left')
    # Count(end <= t)  = searchsorted(e_sorted, t, side='right')
    # Active strictly spanning t (start < t and end > t) = Count(start < t) - Count(end <= t)
    # Valid because every interval satisfies start < end (so start >= t implies end > t).
    started_before = np.searchsorted(s_sorted, spine_utc_ns, side="left")
    ended_by = np.searchsorted(e_sorted, spine_utc_ns, side="right")
    active = started_before - ended_by
    return active.astype("int16")


def build_station_features_for_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Given a contiguous station-hour panel sorted by (station_id, hour_timestamp_utc)
    containing [station_id, hour_timestamp_utc, sessions_count, energy_kwh_arrival_sum,
    valid_dur_session_cnt, valid_plugged_dur_sum, valid_active_chg_sum],
    compute all strictly causal lag, rolling, and station-level expanding features.
    """
    df = panel.copy()
    g = df.groupby("station_id", sort=False)

    # Shifted primary target (strictly past <= t-1)
    s_shift1 = g["sessions_count"].shift(1)
    e_shift1 = g["energy_kwh_arrival_sum"].shift(1)

    # Exact lag features (Req 4)
    df["lag_1h_sessions"] = s_shift1
    df["lag_2h_sessions"] = g["sessions_count"].shift(2)
    df["lag_3h_sessions"] = g["sessions_count"].shift(3)
    df["lag_24h_sessions"] = g["sessions_count"].shift(24)
    df["lag_168h_sessions"] = g["sessions_count"].shift(168)

    # Rolling averages & sums strictly on shifted values [t-W .. t-1]
    # Compute per station cleanly using groupby on shifted series
    df["_s_shift1"] = s_shift1
    df["_e_shift1"] = e_shift1
    gs = df.groupby("station_id", sort=False)

    df["rolling_mean_24h_sessions"] = gs["_s_shift1"].transform(
        lambda x: x.rolling(window=24, min_periods=24).mean()
    )
    df["rolling_sum_24h_sessions"] = gs["_s_shift1"].transform(
        lambda x: x.rolling(window=24, min_periods=24).sum()
    )
    df["rolling_mean_168h_sessions"] = gs["_s_shift1"].transform(
        lambda x: x.rolling(window=168, min_periods=168).mean()
    )
    df["rolling_sum_168h_sessions"] = gs["_s_shift1"].transform(
        lambda x: x.rolling(window=168, min_periods=168).sum()
    )

    # Optional secondary target lags & rolling means (strictly shifted)
    df["lag_1h_energy_kwh"] = e_shift1
    df["lag_24h_energy_kwh"] = g["energy_kwh_arrival_sum"].shift(24)
    df["lag_168h_energy_kwh"] = g["energy_kwh_arrival_sum"].shift(168)
    df["rolling_mean_24h_energy_kwh"] = gs["_e_shift1"].transform(
        lambda x: x.rolling(window=24, min_periods=24).mean()
    )
    df["rolling_mean_168h_energy_kwh"] = gs["_e_shift1"].transform(
        lambda x: x.rolling(window=168, min_periods=168).mean()
    )

    # Station-level strictly causal expanding historical activity & averages (Req 3)
    df["station_cum_sessions_prior"] = gs["_s_shift1"].cumsum().fillna(0.0).astype("int64")
    df["station_expanding_mean_sessions"] = gs["_s_shift1"].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )

    # Station x Hour-of-day strictly causal expanding average (uses only prior days at same clock hour)
    df["station_hour_expanding_mean_sessions"] = (
        df.groupby(["station_id", "hour"], sort=False)["sessions_count"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Hours since last session strictly prior to hour t (causal)
    # At each index i, track last hour index <= i-1 where sessions_count > 0
    df["_row_pos"] = g.cumcount()
    df["_had_session_pos"] = np.where(df["sessions_count"] > 0, df["_row_pos"], np.nan)
    last_pos_prior = df.groupby("station_id", sort=False)["_had_session_pos"].transform(
        lambda x: x.shift(1).ffill()
    )
    df["hours_since_last_session"] = df["_row_pos"] - last_pos_prior

    # Strictly causal historical energy/session & valid duration averages (excluding unreliable duration rows)
    cum_energy_prior = gs["_e_shift1"].cumsum()
    cum_sess_prior_f = df["station_cum_sessions_prior"].astype("float64")
    df["station_hist_mean_energy_per_session"] = np.where(
        cum_sess_prior_f > 0, cum_energy_prior / cum_sess_prior_f, np.nan
    )

    df["_vd_cnt_s1"] = g["valid_dur_session_cnt"].shift(1)
    df["_vp_sum_s1"] = g["valid_plugged_dur_sum"].shift(1)
    df["_va_sum_s1"] = g["valid_active_chg_sum"].shift(1)
    gs2 = df.groupby("station_id", sort=False)
    cum_vd_cnt = gs2["_vd_cnt_s1"].cumsum()
    cum_vp_sum = gs2["_vp_sum_s1"].cumsum()
    cum_va_sum = gs2["_va_sum_s1"].cumsum()

    df["station_hist_mean_plugged_dur_min"] = np.where(cum_vd_cnt > 0, cum_vp_sum / cum_vd_cnt, np.nan)
    df["station_hist_mean_active_chg_min"] = np.where(cum_vd_cnt > 0, cum_va_sum / cum_vd_cnt, np.nan)

    # Clean up temporary helper columns
    df.drop(
        columns=["_s_shift1", "_e_shift1", "_row_pos", "_had_session_pos", "_vd_cnt_s1", "_vp_sum_s1", "_va_sum_s1"],
        inplace=True,
    )
    return df


def main():
    print("Step 1: Verifying input Phase 3 cleaned dataset & raw file hashes...", flush=True)
    raw_sha_before = sha256_file(RAW_ROOT_CSV)
    cleaned_sha_before = sha256_file(CLEANED_CSV)

    req_cols = [
        "station_id",
        "start_datetime_pt",
        "start_datetime_utc",
        "occupancy_start_utc",
        "occupancy_end_utc",
        "active_charging_end_utc",
        "energy_kwh",
        "charging_duration_min",
        "total_duration_min",
        "flag_duration_unreliable",
        "flag_occupancy_interval_unreliable",
    ]
    df_clean = pd.read_csv(CLEANED_CSV, usecols=req_cols, low_memory=False)
    assert len(df_clean) == 259411, f"Expected 259,411 cleaned rows, got {len(df_clean)}"
    for c in req_cols:
        assert c in df_clean.columns, f"Missing required column: {c}"
    print(f"  Verified {len(df_clean):,} rows and {df_clean['station_id'].nunique()} stations.", flush=True)

    # Parse UTC timestamps
    df_clean["start_utc_dt"] = pd.to_datetime(df_clean["start_datetime_utc"])
    df_clean["hour_timestamp_utc"] = df_clean["start_utc_dt"].dt.floor("h")
    df_clean["occ_start_utc_dt"] = pd.to_datetime(df_clean["occupancy_start_utc"])
    df_clean["occ_end_utc_dt"] = pd.to_datetime(df_clean["occupancy_end_utc"])
    df_clean["act_end_utc_dt"] = pd.to_datetime(df_clean["active_charging_end_utc"])

    # Valid duration mask (Req 6: exclude flagged invalid duration values from duration predictors)
    valid_dur_mask = ~df_clean["flag_duration_unreliable"].astype(bool)
    df_clean["valid_dur_flag"] = valid_dur_mask.astype("int32")
    df_clean["valid_plugged_min"] = np.where(valid_dur_mask, df_clean["total_duration_min"], 0.0)
    df_clean["valid_active_min"] = np.where(valid_dur_mask, df_clean["charging_duration_min"], 0.0)

    print("Step 2: Aggregating session arrivals & energy by (station_id, arrival hour)...", flush=True)
    hourly_agg = (
        df_clean.groupby(["station_id", "hour_timestamp_utc"], as_index=False)
        .agg(
            sessions_count=("energy_kwh", "size"),
            energy_kwh_arrival_sum=("energy_kwh", "sum"),
            valid_dur_session_cnt=("valid_dur_flag", "sum"),
            valid_plugged_dur_sum=("valid_plugged_min", "sum"),
            valid_active_chg_sum=("valid_active_min", "sum"),
        )
    )
    assert int(hourly_agg["sessions_count"].sum()) == 259411, "Arrival count conservation failed!"

    print("Step 3: Building contiguous per-station hourly spines [first_hour .. last_hour]...", flush=True)
    # Build per-station contiguous UTC hourly grid from station's first to last recorded arrival hour
    # so we NEVER invent zero-demand periods before a station began operating or after it ceased reporting.
    station_bounds = (
        df_clean.groupby("station_id")["hour_timestamp_utc"]
        .agg(first_hour_utc="min", last_hour_utc="max", total_sessions="size")
        .reset_index()
        .sort_values("station_id")
    )

    # Pre-split reliable occupancy & active-charging interval arrays per station
    valid_occ_df = df_clean[~df_clean["flag_occupancy_interval_unreliable"].astype(bool)]
    valid_act_df = df_clean[(~df_clean["flag_occupancy_interval_unreliable"].astype(bool)) & valid_dur_mask]

    occ_by_st = {
        st: (
            g["occ_start_utc_dt"].astype("int64").to_numpy(),
            g["occ_end_utc_dt"].astype("int64").to_numpy(),
        )
        for st, g in valid_occ_df.groupby("station_id", sort=False)
    }
    act_by_st = {
        st: (
            g["occ_start_utc_dt"].astype("int64").to_numpy(),
            g["act_end_utc_dt"].astype("int64").to_numpy(),
        )
        for st, g in valid_act_df.groupby("station_id", sort=False)
    }

    spine_frames = []
    for row in station_bounds.itertuples(index=False):
        st = row.station_id
        h_range = pd.date_range(row.first_hour_utc, row.last_hour_utc, freq="h")
        h_ns = h_range.astype("int64").to_numpy()
        age_days = ((h_range - row.first_hour_utc).total_seconds() / 86400.0).to_numpy()

        # Empirical prior plugged-in & active-charging counts at instant t:00:00 (strictly start < t < end)
        s_occ, e_occ = occ_by_st.get(st, (np.array([], dtype="int64"), np.array([], dtype="int64")))
        prior_plugged = compute_prior_occupancy_at_hour_starts(h_ns, s_occ, e_occ)

        s_act, e_act = act_by_st.get(st, (np.array([], dtype="int64"), np.array([], dtype="int64")))
        prior_active = compute_prior_occupancy_at_hour_starts(h_ns, s_act, e_act)

        spine_frames.append(
            pd.DataFrame({
                "station_id": st,
                "hour_timestamp_utc": h_range,
                "station_age_days": np.round(age_days, 4),
                "prior_plugged_in_vehicles_at_hour_start": prior_plugged,
                "prior_active_charging_vehicles_at_hour_start": prior_active,
            })
        )

    panel = pd.concat(spine_frames, ignore_index=True)
    print(f"  Built contiguous panel: {len(panel):,} station-hours across {len(station_bounds)} stations.", flush=True)

    # Merge aggregated targets & valid-duration sums into contiguous spine
    panel = panel.merge(hourly_agg, on=["station_id", "hour_timestamp_utc"], how="left")
    panel["sessions_count"] = panel["sessions_count"].fillna(0).astype("int32")
    panel["energy_kwh_arrival_sum"] = panel["energy_kwh_arrival_sum"].fillna(0.0).round(6)
    panel["valid_dur_session_cnt"] = panel["valid_dur_session_cnt"].fillna(0).astype("int32")
    panel["valid_plugged_dur_sum"] = panel["valid_plugged_dur_sum"].fillna(0.0)
    panel["valid_active_chg_sum"] = panel["valid_active_chg_sum"].fillna(0.0)

    print("Step 4: Generating Pacific local time-based & calendar features...", flush=True)
    pt_naive, pt_off = utc_to_pacific_with_offsets(panel["hour_timestamp_utc"])
    off_str = np.where(pt_off == -7, "-07:00", "-08:00")
    panel["hour_timestamp_pt"] = pt_naive.dt.strftime("%Y-%m-%d %H:%M:%S") + off_str

    panel["hour"] = pt_naive.dt.hour.astype("int8")
    panel["day"] = pt_naive.dt.day.astype("int8")
    panel["day_of_week"] = pt_naive.dt.dayofweek.astype("int8")  # 0=Mon .. 6=Sun
    panel["month"] = pt_naive.dt.month.astype("int8")
    panel["year"] = pt_naive.dt.year.astype("int16")
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")

    # US federal holiday & working-day indicator (pure deterministic calendar rules)
    all_years = sorted(panel["year"].unique().tolist())
    fed_holidays = get_us_federal_holidays(all_years)
    pt_dates = pt_naive.dt.date
    panel["is_us_federal_holiday"] = pt_dates.isin(fed_holidays).astype("int8")
    panel["is_working_day"] = ((panel["is_weekend"] == 0) & (panel["is_us_federal_holiday"] == 0)).astype("int8")

    # Cyclical features
    panel["hour_sin"] = np.round(np.sin(2.0 * np.pi * panel["hour"] / 24.0), 6)
    panel["hour_cos"] = np.round(np.cos(2.0 * np.pi * panel["hour"] / 24.0), 6)
    panel["dow_sin"] = np.round(np.sin(2.0 * np.pi * panel["day_of_week"] / 7.0), 6)
    panel["dow_cos"] = np.round(np.cos(2.0 * np.pi * panel["day_of_week"] / 7.0), 6)
    panel["month_sin"] = np.round(np.sin(2.0 * np.pi * (panel["month"] - 1) / 12.0), 6)
    panel["month_cos"] = np.round(np.cos(2.0 * np.pi * (panel["month"] - 1) / 12.0), 6)

    # Policy pricing regime indicator (observed Aug 1, 2017 shift from 0% paid to ~95-99% paid)
    panel["is_paid_era"] = (pt_naive >= pd.Timestamp("2017-08-01 00:00:00")).astype("int8")

    print("Step 5: Computing leakage-safe lag, rolling, and expanding station features...", flush=True)
    # Sort by (station_id, hour_timestamp_utc) before computing group shifts/rollings
    panel = panel.sort_values(["station_id", "hour_timestamp_utc"]).reset_index(drop=True)
    panel = build_station_features_for_panel(panel)

    # Round floating point features cleanly for compact CSV storage
    float_round_cols = [
        "rolling_mean_24h_sessions", "rolling_mean_168h_sessions",
        "rolling_mean_24h_energy_kwh", "rolling_mean_168h_energy_kwh",
        "station_expanding_mean_sessions", "station_hour_expanding_mean_sessions",
        "station_hist_mean_energy_per_session",
        "station_hist_mean_plugged_dur_min", "station_hist_mean_active_chg_min",
        "lag_1h_energy_kwh", "lag_24h_energy_kwh", "lag_168h_energy_kwh",
    ]
    for c in float_round_cols:
        panel[c] = panel[c].round(6)

    # =========================================================================
    # STEP 6: AUTOMATED QUALITY & ANTI-LEAKAGE VERIFICATION SUITE
    # =========================================================================
    print("Step 6: Running automated quality & future-leakage verification checks...", flush=True)
    checks = {}

    # 6a. Missing targets
    checks["zero_missing_primary_target"] = int(panel["sessions_count"].isna().sum()) == 0
    checks["zero_missing_secondary_target"] = int(panel["energy_kwh_arrival_sum"].isna().sum()) == 0

    # 6b. Arrival & energy conservation
    checks["exact_session_arrival_conservation"] = int(panel["sessions_count"].sum()) == 259411
    checks["exact_energy_conservation"] = abs(float(panel["energy_kwh_arrival_sum"].sum()) - float(df_clean["energy_kwh"].sum())) < 0.05

    # 6c. Non-negative demand & occupancy counts
    checks["zero_negative_sessions_count"] = int((panel["sessions_count"] < 0).sum()) == 0
    checks["zero_negative_energy"] = int((panel["energy_kwh_arrival_sum"] < 0).sum()) == 0
    checks["zero_negative_prior_occupancy"] = (
        int((panel["prior_plugged_in_vehicles_at_hour_start"] < 0).sum()) == 0
        and int((panel["prior_active_charging_vehicles_at_hour_start"] < 0).sum()) == 0
    )

    # 6d. Duplicate station-hour records (both UTC and offset-aware Pacific Time)
    checks["zero_duplicate_station_hour_utc"] = int(panel.duplicated(subset=["station_id", "hour_timestamp_utc"]).sum()) == 0
    checks["zero_duplicate_station_hour_pt"] = int(panel.duplicated(subset=["station_id", "hour_timestamp_pt"]).sum()) == 0

    # 6e. Timestamp validity & exact 1-hour contiguity per station
    diffs_sec = panel.groupby("station_id")["hour_timestamp_utc"].diff().dropna().dt.total_seconds()
    checks["exact_1h_contiguity_all_stations"] = bool((diffs_sec == 3600.0).all())
    checks["zero_invalid_timestamps"] = int(panel["hour_timestamp_utc"].isna().sum()) == 0 and int(panel["hour_timestamp_pt"].isna().sum()) == 0

    # 6f. Exact lag & rolling verification across the full dataset
    g_chk = panel.groupby("station_id", sort=False)["sessions_count"]
    for lag_h, col_name in [
        (1, "lag_1h_sessions"), (2, "lag_2h_sessions"), (3, "lag_3h_sessions"),
        (24, "lag_24h_sessions"), (168, "lag_168h_sessions"),
    ]:
        expected = g_chk.shift(lag_h)
        both_valid = expected.notna() & panel[col_name].notna()
        mismatch_cnt = int((expected[both_valid] != panel[col_name][both_valid]).sum())
        na_mismatch = int((expected.isna() != panel[col_name].isna()).sum())
        checks[f"exact_{col_name}"] = (mismatch_cnt == 0 and na_mismatch == 0)

    # Independent manual check of rolling_mean_24h & rolling_mean_168h on a busy station
    sample_st = "PALO ALTO CA / HAMILTON #2"
    sub = panel[panel["station_id"] == sample_st].reset_index(drop=True)
    s_vals = sub["sessions_count"].to_numpy(dtype=float)
    r24_ok = True
    r168_ok = True
    for idx_t in [24, 25, 100, 168, 500, 10000, 50000, len(sub) - 1]:
        exp24 = float(np.mean(s_vals[idx_t - 24 : idx_t]))
        if abs(exp24 - float(sub.loc[idx_t, "rolling_mean_24h_sessions"])) > 1e-5:
            r24_ok = False
        if idx_t >= 168:
            exp168 = float(np.mean(s_vals[idx_t - 168 : idx_t]))
            if abs(exp168 - float(sub.loc[idx_t, "rolling_mean_168h_sessions"])) > 1e-5:
                r168_ok = False
    checks["exact_rolling_24h_verified"] = r24_ok
    checks["exact_rolling_168h_verified"] = r168_ok

    # 6g. Adversarial Future-Data Leakage Perturbation Proof!
    # Mutate targets at t >= t_star on sample station by +9999 and verify ALL engineered features at t <= t_star
    # (including t = t_star itself!) are bit-for-bit unchanged.
    base_cols = [
        "station_id", "hour_timestamp_utc", "hour",
        "sessions_count", "energy_kwh_arrival_sum",
        "valid_dur_session_cnt", "valid_plugged_dur_sum", "valid_active_chg_sum",
    ]
    test_sub = panel[panel["station_id"] == sample_st][base_cols].copy().reset_index(drop=True)
    t_star = 1000
    orig_feats = build_station_features_for_panel(test_sub)
    mut_sub = test_sub.copy()
    mut_sub.loc[t_star:, "sessions_count"] += 9999
    mut_sub.loc[t_star:, "energy_kwh_arrival_sum"] += 9999.0
    mut_sub.loc[t_star:, "valid_dur_session_cnt"] += 99
    mut_sub.loc[t_star:, "valid_plugged_dur_sum"] += 9999.0
    mut_sub.loc[t_star:, "valid_active_chg_sum"] += 9999.0
    mut_feats = build_station_features_for_panel(mut_sub)

    feature_cols_to_audit = [
        c for c in orig_feats.columns
        if c not in [
            "station_id", "hour_timestamp_utc", "hour",
            "sessions_count", "energy_kwh_arrival_sum",
            "valid_dur_session_cnt", "valid_plugged_dur_sum", "valid_active_chg_sum",
        ]
    ]
    leaked_cols = []
    for fc in feature_cols_to_audit:
        s1 = orig_feats.loc[:t_star, fc]
        s2 = mut_feats.loc[:t_star, fc]
        # Equal including NaN positions
        if not s1.equals(s2):
            leaked_cols.append(fc)
    checks["zero_future_leakage_perturbation_proof"] = len(leaked_cols) == 0

    print("  ALL CHECKS:", checks, flush=True)
    assert all(checks.values()), f"FAILED CHECKS: {[k for k, v in checks.items() if not v]}"

    # =========================================================================
    # STEP 7: FINAL COLUMN ORDERING, CHRONOLOGICAL SORT & EXPORT
    # =========================================================================
    print("Step 7: Sorting chronologically by (hour_timestamp_utc, station_id) & exporting CSV...", flush=True)
    final_cols = [
        # Entity & Timestamps
        "station_id",
        "hour_timestamp_pt",
        "hour_timestamp_utc",
        # Targets
        "sessions_count",
        "energy_kwh_arrival_sum",
        # Calendar & Time-based Features
        "hour",
        "day",
        "day_of_week",
        "month",
        "year",
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
        # Station-level & Causal Historical Context
        "station_age_days",
        "station_cum_sessions_prior",
        "station_expanding_mean_sessions",
        "station_hour_expanding_mean_sessions",
        "hours_since_last_session",
        "station_hist_mean_energy_per_session",
        "station_hist_mean_plugged_dur_min",
        "station_hist_mean_active_chg_min",
        # Causal Empirical Pre-Hour Concurrency (excludes unreliable intervals)
        "prior_plugged_in_vehicles_at_hour_start",
        "prior_active_charging_vehicles_at_hour_start",
        # Historical Demand Lags & Rolling Features (Primary Target)
        "lag_1h_sessions",
        "lag_2h_sessions",
        "lag_3h_sessions",
        "lag_24h_sessions",
        "lag_168h_sessions",
        "rolling_mean_24h_sessions",
        "rolling_sum_24h_sessions",
        "rolling_mean_168h_sessions",
        "rolling_sum_168h_sessions",
        # Historical Energy Lags & Rolling Features (Secondary Target)
        "lag_1h_energy_kwh",
        "lag_24h_energy_kwh",
        "lag_168h_energy_kwh",
        "rolling_mean_24h_energy_kwh",
        "rolling_mean_168h_energy_kwh",
    ]

    # Sort strictly chronologically for time-series ML readiness (no random shuffling!)
    panel_out = panel[final_cols].sort_values(["hour_timestamp_utc", "station_id"]).reset_index(drop=True)
    # Format UTC timestamp cleanly
    panel_out["hour_timestamp_utc"] = panel_out["hour_timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")

    panel_out.to_csv(OUTPUT_CSV, index=False)
    out_size_mb = round(os.path.getsize(OUTPUT_CSV) / (1024 ** 2), 2)

    # Re-verify neither raw CSV nor Phase 3 cleaned CSV was touched
    assert sha256_file(RAW_ROOT_CSV) == raw_sha_before, "CRITICAL: Raw CSV changed!"
    assert sha256_file(CLEANED_CSV) == cleaned_sha_before, "CRITICAL: Phase 3 cleaned CSV changed!"

    # Summary statistics for the Phase 4 report
    nonzero_rows = int((panel["sessions_count"] > 0).sum())
    zero_rows = int((panel["sessions_count"] == 0).sum())
    warmup_168h_rows = int(panel["lag_168h_sessions"].isna().sum())
    ready_full_lag_rows = int(panel["lag_168h_sessions"].notna().sum())

    # Year-wise row distribution (helpful for Phase 5 chronological split planning)
    by_year = (
        panel.groupby("year")
        .agg(
            station_hours=("sessions_count", "size"),
            active_stations=("station_id", "nunique"),
            total_sessions=("sessions_count", "sum"),
            total_energy_kwh=("energy_kwh_arrival_sum", "sum"),
            zero_arrival_pct=("sessions_count", lambda s: round(float((s == 0).mean() * 100.0), 2)),
        )
    )
    year_summary = {
        str(int(y)): {
            "station_hours": int(r["station_hours"]),
            "active_stations": int(r["active_stations"]),
            "total_sessions": int(r["total_sessions"]),
            "total_energy_kwh": round(float(r["total_energy_kwh"]), 2),
            "zero_arrival_pct": float(r["zero_arrival_pct"]),
        }
        for y, r in by_year.iterrows()
    }

    # Stations with < 168 hours total span (Dec 2020 Sherman stations)
    short_stations = station_bounds.copy()
    short_stations["span_hours"] = ((short_stations["last_hour_utc"] - short_stations["first_hour_utc"]).dt.total_seconds() / 3600 + 1).astype(int)
    lt_168_st = short_stations[short_stations["span_hours"] < 168][["station_id", "total_sessions", "span_hours"]].to_dict(orient="records")

    summary = {
        "files": {
            "input_cleaned_csv": CLEANED_CSV,
            "output_features_csv": OUTPUT_CSV,
            "output_size_mb": out_size_mb,
            "raw_csv_unchanged": True,
            "phase3_cleaned_csv_unchanged": True,
        },
        "dimensions": {
            "rows": int(len(panel_out)),
            "columns": int(panel_out.shape[1]),
            "unique_stations": int(panel_out["station_id"].nunique()),
            "date_range_pt": [str(panel_out["hour_timestamp_pt"].iloc[0]), str(panel_out["hour_timestamp_pt"].iloc[-1])],
            "date_range_utc": [str(panel_out["hour_timestamp_utc"].iloc[0]), str(panel_out["hour_timestamp_utc"].iloc[-1])],
            "column_names": final_cols,
        },
        "target_stats": {
            "sessions_count_sum": int(panel["sessions_count"].sum()),
            "sessions_count_mean": round(float(panel["sessions_count"].mean()), 4),
            "sessions_count_std": round(float(panel["sessions_count"].std()), 4),
            "sessions_count_max": int(panel["sessions_count"].max()),
            "sessions_count_value_counts": {str(int(k)): int(v) for k, v in panel["sessions_count"].value_counts().sort_index().items()},
            "zero_arrival_station_hours": zero_rows,
            "zero_arrival_pct": round(zero_rows / len(panel) * 100.0, 2),
            "nonzero_arrival_station_hours": nonzero_rows,
            "nonzero_arrival_pct": round(nonzero_rows / len(panel) * 100.0, 2),
            "energy_kwh_sum": round(float(panel["energy_kwh_arrival_sum"].sum()), 2),
            "energy_kwh_mean": round(float(panel["energy_kwh_arrival_sum"].mean()), 4),
            "energy_kwh_max": round(float(panel["energy_kwh_arrival_sum"].max()), 4),
        },
        "occupancy_feature_stats": {
            "prior_plugged_max": int(panel["prior_plugged_in_vehicles_at_hour_start"].max()),
            "prior_plugged_mean": round(float(panel["prior_plugged_in_vehicles_at_hour_start"].mean()), 4),
            "prior_active_max": int(panel["prior_active_charging_vehicles_at_hour_start"].max()),
            "prior_active_mean": round(float(panel["prior_active_charging_vehicles_at_hour_start"].mean()), 4),
            "excluded_unreliable_occupancy_sessions": int(df_clean["flag_occupancy_interval_unreliable"].sum()),
            "excluded_unreliable_duration_sessions": int(df_clean["flag_duration_unreliable"].sum()),
        },
        "missing_values_per_column": {c: int(panel_out[c].isna().sum()) for c in final_cols if panel_out[c].isna().sum() > 0},
        "warmup_info": {
            "rows_missing_168h_lag": warmup_168h_rows,
            "rows_with_complete_168h_lags": ready_full_lag_rows,
            "pct_complete_168h_lags": round(ready_full_lag_rows / len(panel) * 100.0, 2),
            "stations_with_lt_168h_span": lt_168_st,
        },
        "by_year": year_summary,
        "quality_checks": checks,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"SUCCESS! Saved {len(panel_out):,} rows x {panel_out.shape[1]} cols to {OUTPUT_CSV} ({out_size_mb} MB).", flush=True)


if __name__ == "__main__":
    main()
