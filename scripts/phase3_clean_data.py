import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

WORKSPACE = r"c:\Users\harsh\OneDrive\Desktop\ML"
RAW_ROOT_CSV = os.path.join(WORKSPACE, "EVChargingStationUsage.csv")
RAW_COPY_DIR = os.path.join(WORKSPACE, "dataset", "raw")
RAW_COPY_CSV = os.path.join(RAW_COPY_DIR, "EVChargingStationUsage.csv")
PROCESSED_DIR = os.path.join(WORKSPACE, "dataset", "processed")
CLEANED_CSV = os.path.join(PROCESSED_DIR, "ev_charging_cleaned.csv")
SUMMARY_JSON = r"C:\Users\harsh\.gemini\antigravity\brain\7aee0d63-4143-475b-8d08-cc48808ea73c\scratch\phase3_cleaning_summary.json"

EXCEL_EPOCH = datetime(1899, 12, 30)
TZ_OFFSETS = {"PDT": -7, "PST": -8, "UTC": 0}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hms_to_seconds(series: pd.Series) -> pd.Series:
    """Parse hh:mm:ss duration strings (including >= 24h) into float seconds."""
    parts = series.astype(str).str.strip().str.extract(r"^(-?)(\d+):(\d{1,2}):(\d{1,2})$")
    sign = np.where(parts[0] == "-", -1.0, 1.0)
    secs = sign * (
        pd.to_numeric(parts[1], errors="coerce") * 3600.0
        + pd.to_numeric(parts[2], errors="coerce") * 60.0
        + pd.to_numeric(parts[3], errors="coerce")
    )
    return pd.Series(secs, index=series.index)


def us_pacific_offset_from_utc(dt_utc: datetime) -> int:
    """Return Pacific Time offset (-7 for PDT, -8 for PST) given a naive UTC datetime (US DST rules 2011-2021)."""
    y = dt_utc.year
    # Second Sunday in March at 02:00 PST = 10:00 UTC
    mar1 = datetime(y, 3, 1)
    first_sun_mar = 1 + (6 - mar1.weekday()) % 7
    dst_start_utc = datetime(y, 3, first_sun_mar + 7, 10, 0)
    # First Sunday in November at 02:00 PDT = 09:00 UTC
    nov1 = datetime(y, 11, 1)
    first_sun_nov = 1 + (6 - nov1.weekday()) % 7
    dst_end_utc = datetime(y, 11, first_sun_nov, 9, 0)
    return -7 if dst_start_utc <= dt_utc < dst_end_utc else -8


def utc_series_to_pacific(utc_ser: pd.Series) -> pd.Series:
    """Vectorized conversion of naive UTC Timestamps to Pacific local wall-clock Timestamps."""
    # Precompute DST boundaries per year present in the series
    out = utc_ser.copy()
    valid = utc_ser.notna()
    if not valid.any():
        return out
    s_valid = utc_ser[valid]
    years = s_valid.dt.year.unique()
    offsets_hr = pd.Series(-8, index=s_valid.index, dtype="int64")
    for y in years:
        y_int = int(y)
        mar1 = datetime(y_int, 3, 1)
        first_sun_mar = 1 + (6 - mar1.weekday()) % 7
        dst_start_utc = pd.Timestamp(datetime(y_int, 3, first_sun_mar + 7, 10, 0))
        nov1 = datetime(y_int, 11, 1)
        first_sun_nov = 1 + (6 - nov1.weekday()) % 7
        dst_end_utc = pd.Timestamp(datetime(y_int, 11, first_sun_nov, 9, 0))
        in_dst = (s_valid >= dst_start_utc) & (s_valid < dst_end_utc)
        offsets_hr.loc[in_dst] = -7
    out.loc[valid] = s_valid + pd.to_timedelta(offsets_hr, unit="h")
    return out


def count_same_mac_port_overlaps(starts_utc: pd.Series, ends_utc: pd.Series, macs: pd.Series, ports: pd.Series) -> dict:
    tmp = pd.DataFrame({"s": starts_utc, "e": ends_utc, "m": macs, "p": ports}).dropna()
    tmp = tmp[tmp["e"] > tmp["s"]].sort_values(["m", "p", "s", "e"])
    ov_all = ov_gt2m = ov_gt10m = 0
    for (_, _), g in tmp.groupby(["m", "p"], sort=False):
        prev_e = None
        for sv, ev in zip(g["s"].values, g["e"].values):
            if prev_e is not None and sv < prev_e:
                ov_all += 1
                diff_s = (prev_e - sv) / np.timedelta64(1, "s")
                if diff_s > 120:
                    ov_gt2m += 1
                if diff_s > 600:
                    ov_gt10m += 1
            if prev_e is None or ev > prev_e:
                prev_e = ev
    return {"all_overlaps": int(ov_all), "overlaps_gt_2min": int(ov_gt2m), "overlaps_gt_10min": int(ov_gt10m)}


def max_concurrency_for_intervals(starts_utc: pd.Series, ends_utc: pd.Series) -> int:
    valid = starts_utc.notna() & ends_utc.notna() & (ends_utc > starts_utc)
    if not valid.any():
        return 0
    ev = []
    for s_t, e_t in zip(starts_utc[valid].values, ends_utc[valid].values):
        ev.append((s_t, 1))
        ev.append((e_t, -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in ev:
        cur += d
        if cur > mx:
            mx = cur
    return int(mx)


def main():
    os.makedirs(RAW_COPY_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Preserve original dataset byte-for-byte
    print("Step 1: Verifying and preserving original raw CSV...", flush=True)
    raw_sha_before = sha256_file(RAW_ROOT_CSV)
    if not os.path.exists(RAW_COPY_CSV):
        shutil.copy2(RAW_ROOT_CSV, RAW_COPY_CSV)
    raw_copy_sha = sha256_file(RAW_COPY_CSV)
    assert raw_sha_before == raw_copy_sha, "Raw copy SHA-256 mismatch!"
    print(f"  Raw CSV verified & copied unchanged to {RAW_COPY_CSV} (SHA-256: {raw_sha_before[:16]}...)", flush=True)

    # Load raw CSV strictly as strings/original values first to avoid silent coercions,
    # keeping exact raw columns intact.
    df_raw = pd.read_csv(RAW_ROOT_CSV, low_memory=False)
    raw_rows, raw_cols = df_raw.shape
    raw_missing_per_col = {col: int(df_raw[col].isna().sum()) for col in df_raw.columns}
    print(f"  Loaded raw dataset: {raw_rows:,} rows x {raw_cols} columns.", flush=True)

    # 2. Remove ONLY confirmed exact full-row duplicates (4 rows)
    print("Step 2: Removing confirmed exact full-row duplicates...", flush=True)
    exact_dup_mask = df_raw.duplicated(keep="first")
    exact_dups_removed_df = df_raw[exact_dup_mask][
        ["Station Name", "MAC Address", "Port Number", "Start Date", "End Date", "Energy (kWh)", "Plug In Event Id"]
    ].to_dict(orient="records")
    n_exact_dups = int(exact_dup_mask.sum())
    df = df_raw[~exact_dup_mask].copy().reset_index(drop=True)
    cleaned_rows = len(df)
    print(f"  Removed {n_exact_dups} exact full-row duplicate records ({raw_rows:,} -> {cleaned_rows:,}).", flush=True)

    # 3. Standardize Identifiers (without overwriting original columns)
    print("Step 3: Standardizing identifiers (MAC Address, Station Name, Org Name)...", flush=True)
    # Org Name: strip trailing whitespace
    org_ws_fixed = int((df["Org Name"].astype(str) != df["Org Name"].astype(str).str.strip()).sum())
    df["org_name_clean"] = df["Org Name"].astype(str).str.strip()

    # MAC Address: uppercase hex without colons (preserving original MAC Address column)
    raw_unique_macs = int(df["MAC Address"].nunique(dropna=True))
    df["mac_address_clean"] = (
        df["MAC Address"].astype(str).str.strip().str.replace(":", "", regex=False).str.upper()
    )
    clean_unique_macs = int(df["mac_address_clean"].nunique(dropna=True))
    macs_with_colons_count = int(df["MAC Address"].astype(str).str.contains(":", na=False).sum())

    # Station Name: fix confirmed hardware-verified typo "PALO ALTO CA / BRYANT # 1" -> "PALO ALTO CA / BRYANT #1"
    # Verify hardware match explicitly before correcting
    bryant_typo_mask = (
        (df["Station Name"].astype(str).str.strip() == "PALO ALTO CA / BRYANT # 1")
        & (df["mac_address_clean"] == "0024B1000003CF38")
        & (pd.to_numeric(df["EVSE ID"], errors="coerce") == 5358941)
    )
    raw_unique_stations = int(df["Station Name"].nunique(dropna=True))
    df["station_id"] = df["Station Name"].astype(str).str.strip()
    df.loc[bryant_typo_mask, "station_id"] = "PALO ALTO CA / BRYANT #1"
    df["flag_station_name_standardized"] = bryant_typo_mask
    clean_unique_stations = int(df["station_id"].nunique(dropna=True))
    print(
        f"  Stations: {raw_unique_stations} -> {clean_unique_stations} ({int(bryant_typo_mask.sum())} 'BRYANT # 1' rows fixed); "
        f"MACs: {raw_unique_macs} -> {clean_unique_macs} physical units.",
        flush=True,
    )

    # 4. Handle Missing Values — Unambiguous Port Type recovery only
    print("Step 4: Recovering unambiguous missing Port Type values from Plug Type...", flush=True)
    # Verify deterministic Plug Type -> Port Type mapping on non-null rows
    plug_to_port_check = (
        df.dropna(subset=["Port Type", "Plug Type"])
        .groupby("Plug Type")["Port Type"]
        .unique()
        .apply(list)
        .to_dict()
    )
    assert plug_to_port_check == {"J1772": ["Level 2"], "NEMA 5-20R": ["Level 1"]}, (
        f"Unexpected Plug Type -> Port Type mapping: {plug_to_port_check}"
    )
    port_missing_mask = df["Port Type"].isna()
    plug_map = {"J1772": "Level 2", "NEMA 5-20R": "Level 1"}
    df["port_type_clean"] = df["Port Type"]
    df.loc[port_missing_mask, "port_type_clean"] = df.loc[port_missing_mask, "Plug Type"].map(plug_map)
    df["flag_port_type_recovered"] = port_missing_mask & df["port_type_clean"].notna()
    port_recovered_breakdown = df.loc[df["flag_port_type_recovered"], "port_type_clean"].value_counts().to_dict()
    print(f"  Recovered {int(df['flag_port_type_recovered'].sum())} missing Port Type rows: {port_recovered_breakdown}", flush=True)

    # 5. Correct Timestamp Issues (preserving original Start Date, End Date, Transaction Date)
    print("Step 5: Parsing & normalizing timestamps to Pacific Time and UTC...", flush=True)
    start_tz = df["Start Time Zone"].astype(str).str.strip()
    end_tz = df["End Time Zone"].astype(str).str.strip()

    # Parse Start Date
    start_naive = pd.to_datetime(df["Start Date"], errors="coerce")
    flag_start_utc = start_tz == "UTC"
    start_utc = start_naive - pd.to_timedelta(start_tz.map(TZ_OFFSETS), unit="h")
    start_pt = utc_series_to_pacific(start_utc)

    # Parse End Date (including 40 Excel serial floats)
    end_num = pd.to_numeric(df["End Date"], errors="coerce")
    flag_excel_end = end_num.notna()
    end_naive = pd.to_datetime(df["End Date"], errors="coerce")
    if flag_excel_end.any():
        conv_end = (pd.Timestamp(EXCEL_EPOCH) + pd.to_timedelta(end_num[flag_excel_end], unit="D")).dt.round("s")
        end_naive.loc[flag_excel_end] = conv_end.astype(end_naive.dtype)

    flag_end_utc = end_tz == "UTC"
    end_utc = end_naive - pd.to_timedelta(end_tz.map(TZ_OFFSETS), unit="h")
    end_pt = utc_series_to_pacific(end_utc)

    # Parse Transaction Date (Pacific Time) (including 40 Excel serial floats)
    tx_num = pd.to_numeric(df["Transaction Date (Pacific Time)"], errors="coerce")
    flag_excel_tx = tx_num.notna()
    tx_pt = pd.to_datetime(df["Transaction Date (Pacific Time)"], errors="coerce")
    if flag_excel_tx.any():
        conv_tx = (pd.Timestamp(EXCEL_EPOCH) + pd.to_timedelta(tx_num[flag_excel_tx], unit="D")).dt.round("s")
        tx_pt.loc[flag_excel_tx] = conv_tx.astype(tx_pt.dtype)

    df["start_datetime_pt"] = start_pt
    df["end_datetime_pt"] = end_pt
    df["transaction_datetime_pt"] = tx_pt
    df["start_datetime_utc"] = start_utc
    df["end_datetime_utc"] = end_utc
    df["flag_start_utc_converted"] = flag_start_utc
    df["flag_end_utc_converted"] = flag_end_utc
    df["flag_excel_serial_date_fixed"] = flag_excel_end | flag_excel_tx

    # 6. Parse Durations, Energy & Flag Anomalies
    print("Step 6: Validating durations, energy, implied power, and End Date mismatches...", flush=True)
    td_sec = parse_hms_to_seconds(df["Total Duration (hh:mm:ss)"])
    ct_sec = parse_hms_to_seconds(df["Charging Time (hh:mm:ss)"])
    energy_kwh = pd.to_numeric(df["Energy (kWh)"], errors="coerce")
    fee_usd = pd.to_numeric(df["Fee"], errors="coerce")

    df["total_duration_sec"] = td_sec
    df["total_duration_min"] = (td_sec / 60.0).round(4)
    df["charging_duration_sec"] = ct_sec
    df["charging_duration_min"] = (ct_sec / 60.0).round(4)
    df["energy_kwh"] = energy_kwh

    # Charging Time > Total Duration anomaly (45 rows: 40 modulo-24h rollover + 5 <=60s clock sync)
    flag_chg_gt_total = ct_sec > td_sec
    flag_chg_gt_total_rollover = (ct_sec - td_sec) > 60.0
    flag_chg_gt_total_minor = flag_chg_gt_total & (~flag_chg_gt_total_rollover)

    # Idle duration: ONLY compute where Charging Time <= Total Duration (avoid invalid negative idle times)
    idle_min = np.where(~flag_chg_gt_total, (td_sec - ct_sec) / 60.0, np.nan)
    df["idle_duration_min"] = pd.Series(idle_min, index=df.index).round(4)

    # Implied average charging power (kW) for charging time >= 60s
    implied_kw = np.where(ct_sec >= 60.0, energy_kwh / (ct_sec / 3600.0), np.nan)
    df["implied_avg_power_kw"] = pd.Series(implied_kw, index=df.index).round(4)
    # Also compute unconstrained implied power check for <60s
    implied_kw_any = np.where(ct_sec > 0.0, energy_kwh / (ct_sec / 3600.0), np.nan)

    # Anomaly flags
    flag_extreme_total_24h = td_sec > 24 * 3600
    flag_extreme_total_48h = td_sec > 48 * 3600
    flag_extreme_chg_12h = ct_sec > 12 * 3600
    flag_implausible_power = pd.Series(implied_kw_any > 20.0, index=df.index)
    flag_implausible_power_ge1m = pd.Series(implied_kw > 20.0, index=df.index)

    # End Date vs (Start Date + Total Duration) mismatch check
    utc_diff_sec = (end_utc - start_utc).dt.total_seconds()
    mismatch_sec = utc_diff_sec - td_sec
    flag_end_mismatch_5m = mismatch_sec.abs() > 300.0
    flag_end_mismatch_1h = mismatch_sec.abs() > 3600.0
    flag_end_le_start = utc_diff_sec <= 0.0

    df["flag_charging_gt_total_duration"] = flag_chg_gt_total
    df["flag_duration_rollover_gt_60s"] = flag_chg_gt_total_rollover
    df["flag_extreme_total_duration_gt_24h"] = flag_extreme_total_24h
    df["flag_extreme_charging_time_gt_12h"] = flag_extreme_chg_12h
    df["flag_implausible_power_gt_20kw"] = flag_implausible_power
    df["flag_end_date_mismatch_gt_5min"] = flag_end_mismatch_5m
    df["flag_end_date_mismatch_gt_1h"] = flag_end_mismatch_1h
    df["flag_duration_unreliable"] = flag_chg_gt_total | flag_extreme_total_24h

    # 7. Derive Reliable Occupancy & Active Charging Intervals (Req 8)
    print("Step 7: Constructing validated occupancy and active-charging intervals...", flush=True)
    # Rule for physical occupancy end time:
    # - Use Start_UTC + Total_Duration when End Date has a batch-upload/zero-span mismatch (>5 min or End <= Start)
    #   AND Total Duration itself did NOT suffer modulo-24h rollover (not flag_chg_gt_total_rollover).
    # - Otherwise retain corrected End_UTC (which already fixed Excel serials and UTC timezone).
    use_derived_occ_end = (flag_end_mismatch_5m | flag_end_le_start) & (~flag_chg_gt_total_rollover)

    occ_start_utc = start_utc.copy()
    occ_start_pt = start_pt.copy()
    occ_end_utc = end_utc.copy()
    derived_occ = (
        start_utc.loc[use_derived_occ_end] + pd.to_timedelta(td_sec.loc[use_derived_occ_end], unit="s")
    ).dt.round("s").astype(occ_end_utc.dtype)
    occ_end_utc.loc[use_derived_occ_end] = derived_occ
    occ_end_pt = utc_series_to_pacific(occ_end_utc)

    # Audit source column for occupancy_end
    occ_source = pd.Series("original_end_date", index=df.index)
    occ_source.loc[flag_excel_end] = "excel_serial_converted"
    occ_source.loc[flag_end_utc] = "utc_converted"
    occ_source.loc[flag_chg_gt_total_rollover] = "original_end_date_duration_rollover"
    occ_source.loc[use_derived_occ_end] = "derived_start_plus_total_duration"

    # Active charging interval (using Start + Charging Time)
    act_chg_end_utc = start_utc + pd.to_timedelta(ct_sec, unit="s")
    act_chg_end_pt = utc_series_to_pacific(act_chg_end_utc)

    df["occupancy_start_pt"] = occ_start_pt
    df["occupancy_end_pt"] = occ_end_pt
    df["occupancy_start_utc"] = occ_start_utc
    df["occupancy_end_utc"] = occ_end_utc
    df["occupancy_end_source"] = occ_source
    df["flag_occupancy_interval_corrected"] = use_derived_occ_end
    df["flag_occupancy_interval_unreliable"] = flag_chg_gt_total_rollover | flag_extreme_total_48h
    df["active_charging_end_pt"] = act_chg_end_pt
    df["active_charging_end_utc"] = act_chg_end_utc

    # 8. Basic Temporal Feature Preparation (Req 7 — strictly no target/lag features)
    print("Step 8: Preparing basic temporal features for demand modeling...", flush=True)
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["start_date"] = start_pt.dt.strftime("%Y-%m-%d")
    df["hour_of_day"] = start_pt.dt.hour.astype("int64")
    df["day_of_week"] = start_pt.dt.dayofweek.astype("int64")  # 0=Mon .. 6=Sun
    df["day_of_week_name"] = df["day_of_week"].map(dict(enumerate(dow_names)))
    df["month"] = start_pt.dt.month.astype("int64")
    df["year"] = start_pt.dt.year.astype("int64")
    df["is_weekend"] = df["day_of_week"] >= 5

    # 9. Post-Cleaning Quality Checks (Req 9)
    print("Step 9: Running post-cleaning quality checks & concurrency validation...", flush=True)
    occ_interval_sec = (df["occupancy_end_utc"] - df["occupancy_start_utc"]).dt.total_seconds()
    act_interval_sec = (df["active_charging_end_utc"] - df["occupancy_start_utc"]).dt.total_seconds()

    # Concurrency comparison: raw corrected End_UTC vs validated occupancy_end_utc
    ov_before = count_same_mac_port_overlaps(start_utc, end_utc, df["mac_address_clean"], df["Port Number"])
    ov_after = count_same_mac_port_overlaps(occ_start_utc, occ_end_utc, df["mac_address_clean"], df["Port Number"])

    mpl4_mask = df["station_id"] == "PALO ALTO CA / MPL #4"
    mpl4_conc_before = max_concurrency_for_intervals(start_utc[mpl4_mask], end_utc[mpl4_mask])
    mpl4_conc_after = max_concurrency_for_intervals(occ_start_utc[mpl4_mask], occ_end_utc[mpl4_mask])
    city_conc_before = max_concurrency_for_intervals(start_utc, end_utc)
    city_conc_after = max_concurrency_for_intervals(occ_start_utc, occ_end_utc)

    # Check remaining exact duplicates & key integrity
    remaining_exact_dups = int(df.duplicated().sum())
    remaining_orig_33_dups = int(df[df_raw.columns].duplicated().sum())

    # Format datetime columns cleanly as ISO strings for CSV export
    dt_cols = [
        "start_datetime_pt", "end_datetime_pt", "transaction_datetime_pt",
        "start_datetime_utc", "end_datetime_utc",
        "occupancy_start_pt", "occupancy_end_pt",
        "occupancy_start_utc", "occupancy_end_utc",
        "active_charging_end_pt", "active_charging_end_utc",
    ]
    df_export = df.copy()
    for c in dt_cols:
        df_export[c] = df_export[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Step 10: Writing cleaned CSV to {CLEANED_CSV}...", flush=True)
    df_export.to_csv(CLEANED_CSV, index=False)
    cleaned_size_mb = round(os.path.getsize(CLEANED_CSV) / (1024 ** 2), 2)

    # Re-verify original CSV was NEVER modified
    raw_sha_after = sha256_file(RAW_ROOT_CSV)
    assert raw_sha_before == raw_sha_after, "CRITICAL: Raw CSV hash changed!"

    # Missing values table for all columns (original 33 + new derived columns)
    cleaned_missing_orig = {col: int(df[col].isna().sum()) for col in df_raw.columns}
    cleaned_missing_new = {col: int(df[col].isna().sum()) for col in df.columns if col not in df_raw.columns}

    summary = {
        "files": {
            "raw_root_csv": RAW_ROOT_CSV,
            "raw_copy_csv": RAW_COPY_CSV,
            "raw_sha256": raw_sha_after,
            "raw_unchanged_verified": raw_sha_before == raw_sha_after == raw_copy_sha,
            "cleaned_csv": CLEANED_CSV,
            "cleaned_size_mb": cleaned_size_mb,
        },
        "dimensions": {
            "raw_rows": int(raw_rows),
            "raw_columns": int(raw_cols),
            "cleaned_rows": int(cleaned_rows),
            "cleaned_columns": int(df.shape[1]),
            "new_columns_added": int(df.shape[1] - raw_cols),
            "rows_removed": int(raw_rows - cleaned_rows),
            "new_column_names": [c for c in df.columns if c not in df_raw.columns],
        },
        "duplicates": {
            "exact_full_row_duplicates_removed": n_exact_dups,
            "removed_duplicate_records": exact_dups_removed_df,
            "remaining_exact_duplicates_all_cols": remaining_exact_dups,
            "remaining_exact_duplicates_orig_33_cols": remaining_orig_33_dups,
        },
        "missing_values": {
            "raw_missing_per_col": raw_missing_per_col,
            "cleaned_missing_orig_cols": cleaned_missing_orig,
            "cleaned_missing_new_cols": cleaned_missing_new,
            "port_type_recovered_count": int(df["flag_port_type_recovered"].sum()),
            "port_type_recovered_breakdown": port_recovered_breakdown,
            "port_type_clean_remaining_missing": int(df["port_type_clean"].isna().sum()),
        },
        "identifiers": {
            "raw_unique_stations": raw_unique_stations,
            "cleaned_unique_stations": clean_unique_stations,
            "bryant_typo_rows_fixed": int(bryant_typo_mask.sum()),
            "raw_unique_macs": raw_unique_macs,
            "cleaned_unique_physical_macs": clean_unique_macs,
            "rows_with_colons_in_raw_mac": macs_with_colons_count,
            "org_name_trailing_ws_fixed": org_ws_fixed,
        },
        "timestamps": {
            "start_utc_converted_count": int(flag_start_utc.sum()),
            "end_utc_converted_count": int(flag_end_utc.sum()),
            "total_utc_records_corrected": int((flag_start_utc | flag_end_utc).sum()),
            "excel_serial_end_date_fixed": int(flag_excel_end.sum()),
            "excel_serial_tx_date_fixed": int(flag_excel_tx.sum()),
            "start_datetime_pt_missing": int(df["start_datetime_pt"].isna().sum()),
            "end_datetime_pt_missing": int(df["end_datetime_pt"].isna().sum()),
            "transaction_datetime_pt_missing": int(df["transaction_datetime_pt"].isna().sum()),
            "start_pt_range": [str(df["start_datetime_pt"].min()), str(df["start_datetime_pt"].max())],
            "end_pt_range": [str(df["end_datetime_pt"].min()), str(df["end_datetime_pt"].max())],
            "end_before_start_utc_count": int((utc_diff_sec < 0).sum()),
            "end_equals_start_utc_count": int((utc_diff_sec == 0).sum()),
            "end_date_mismatch_gt_5min_count": int(flag_end_mismatch_5m.sum()),
            "end_date_mismatch_gt_1h_count": int(flag_end_mismatch_1h.sum()),
        },
        "durations_and_energy": {
            "total_duration_negative_or_zero": int((td_sec <= 0).sum()),
            "charging_duration_negative_or_zero": int((ct_sec <= 0).sum()),
            "charging_gt_total_duration_total": int(flag_chg_gt_total.sum()),
            "charging_gt_total_rollover_gt_60s": int(flag_chg_gt_total_rollover.sum()),
            "charging_gt_total_minor_le_60s": int(flag_chg_gt_total_minor.sum()),
            "idle_duration_negative_remaining": int((df["idle_duration_min"] < 0).sum()),
            "idle_duration_set_nan_count": int(df["idle_duration_min"].isna().sum()),
            "extreme_total_duration_gt_24h": int(flag_extreme_total_24h.sum()),
            "extreme_total_duration_gt_48h": int(flag_extreme_total_48h.sum()),
            "extreme_charging_time_gt_12h": int(flag_extreme_chg_12h.sum()),
            "energy_missing": int(df["energy_kwh"].isna().sum()),
            "energy_negative_or_zero": int((df["energy_kwh"] <= 0).sum()),
            "energy_min_max_sum": [
                round(float(df["energy_kwh"].min()), 4),
                round(float(df["energy_kwh"].max()), 4),
                round(float(df["energy_kwh"].sum()), 2),
            ],
            "implausible_power_gt_20kw_all": int(flag_implausible_power.sum()),
            "implausible_power_gt_20kw_ge1m": int(flag_implausible_power_ge1m.sum()),
        },
        "occupancy": {
            "occupancy_end_source_counts": df["occupancy_end_source"].value_counts().to_dict(),
            "occupancy_interval_corrected_count": int(df["flag_occupancy_interval_corrected"].sum()),
            "occupancy_interval_unreliable_count": int(df["flag_occupancy_interval_unreliable"].sum()),
            "occupancy_interval_le_0_remaining": int((occ_interval_sec <= 0).sum()),
            "active_charging_interval_le_0_remaining": int((act_interval_sec <= 0).sum()),
            "same_mac_port_overlaps_before_correction": ov_before,
            "same_mac_port_overlaps_after_correction": ov_after,
            "mpl4_max_concurrency_before": mpl4_conc_before,
            "mpl4_max_concurrency_after": mpl4_conc_after,
            "city_max_concurrency_before": city_conc_before,
            "city_max_concurrency_after": city_conc_after,
        },
        "quality_checks_passed": {
            "raw_file_untouched": raw_sha_before == raw_sha_after,
            "zero_exact_duplicates": remaining_exact_dups == 0 and remaining_orig_33_dups == 0,
            "zero_unparseable_start_or_end": int(df["start_datetime_pt"].isna().sum()) == 0 and int(df["end_datetime_pt"].isna().sum()) == 0,
            "zero_negative_or_zero_occupancy_intervals": int((occ_interval_sec <= 0).sum()) == 0,
            "zero_negative_idle_durations": int((df["idle_duration_min"] < 0).sum()) == 0,
            "zero_missing_port_type_clean": int(df["port_type_clean"].isna().sum()) == 0,
            "zero_negative_or_missing_energy": int(df["energy_kwh"].isna().sum()) == 0 and int((df["energy_kwh"] <= 0).sum()) == 0,
            "station_count_46": clean_unique_stations == 46,
            "physical_mac_count_68": clean_unique_macs == 68,
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("ALL QUALITY CHECKS:", summary["quality_checks_passed"], flush=True)
    assert all(summary["quality_checks_passed"].values()), f"Some quality checks failed: {summary['quality_checks_passed']}"
    print(f"SUCCESS! Cleaned dataset saved to {CLEANED_CSV} ({cleaned_size_mb} MB).", flush=True)


if __name__ == "__main__":
    main()
