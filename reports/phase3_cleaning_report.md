# Phase 3: Data Cleaning and Preprocessing Report

**Project:** Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations  
**Raw Source (Untouched):** `EVChargingStationUsage.csv` (and unchanged archival copy at `dataset/raw/EVChargingStationUsage.csv`)  
**Cleaned Dataset Output:** `dataset/processed/ev_charging_cleaned.csv`  
**Cleaning Pipeline Script:** `scripts/phase3_clean_data.py`

---

## 1. Executive Before-and-After Summary

| Metric | Raw Dataset (`EVChargingStationUsage.csv`) | Cleaned Dataset (`ev_charging_cleaned.csv`) | Change / Treatment |
| :--- | --: | --: | :--- |
| **Total Rows** | 259,415 | **259,411** | **−4 rows** (only confirmed exact full-row duplicates removed) |
| **Total Columns** | 33 | **78** | **+45 columns** (all 33 original columns preserved unmodified + standardized, timestamp, occupancy, temporal, and flag columns) |
| **File Size** | 81.49 MB | **202.89 MB** | Includes preserved original columns + ISO Pacific/UTC timestamps & audit flags |
| **Raw File Integrity (SHA-256)** | `e65f5f5d3861cdd6...` | `e65f5f5d3861cdd6...` | **100% identical** on root CSV and `dataset/raw/EVChargingStationUsage.csv` |
| **Exact Full-Row Duplicates** | 4 extra rows (8 involved) | **0** | Verified across both all 78 columns and original 33 columns |
| **Unique Station Identifiers** | 47 (`Station Name`) | **46** (`station_id`) | 13 hardware-verified `BRYANT # 1` typo rows standardized to `BRYANT #1` |
| **Unique Physical MACs** | 83 raw strings | **68** (`mac_address_clean`) | 15 MACs standardized across colon / non-colon formats; raw column preserved |
| **Missing `Port Type`** | 9 | **0** (`port_type_clean`) | 9 unambiguously recovered from `Plug Type` (7 `Level 2`, 2 `Level 1`) |
| **Unparseable `Start` / `End` Dates** | 0 Start / 40 End / 40 Tx | **0** Start / **0** End / **0** Tx | All 40 Excel serial floats recovered (max error ≤ 51s vs. `Start + Duration`) |
| **UTC-Labeled Timestamps** | 32 (31 Start, 1 End) | **0** in `*_pt` columns | All 32 converted accurately to Pacific local time (`PST`/`PDT`) & UTC |
| **Naive `End < Start` Errors** | 31 | **0** | Eliminated via proper timezone normalization |
| **Zero / Negative Occupancy Intervals** | 12 (`End == Start`) | **0** | All 259,411 `occupancy_end_utc > occupancy_start_utc` |
| **Same-`(MAC, Port)` Overlaps (>10m)** | 234 | **55** | **−76.5% reduction** via `Start + Total Duration` occupancy repair |
| **Invalid Negative Idle Durations** | 45 (`Charging > Total`) | **0** (`45` set to `NaN`) | Flagged & excluded from derived `idle_duration_min`; sessions retained |
| **Total Energy Delivered (kWh)** | 2,216,513.60 | **2,216,493.71** | −19.89 kWh (exact sum of the 4 removed duplicate rows) |

---

## 2. Preservation of Original Dataset

- The original workspace file `EVChargingStationUsage.csv` was opened strictly read-only and verified unchanged before and after script execution.
- An identical byte-for-byte copy was saved to `dataset/raw/EVChargingStationUsage.csv`.
- **SHA-256 Verification:**
  - Root `EVChargingStationUsage.csv` (before & after): `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`
  - Archival `dataset/raw/EVChargingStationUsage.csv`: `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`
- Inside `dataset/processed/ev_charging_cleaned.csv`, **all 33 original raw columns are retained untouched**, so no original timestamp, identifier, duration, or categorical field was silently overwritten.

---

## 3. Duplicate Record Removal

Only the **4 confirmed exact full-row duplicate records** (identical across all 33 columns) were removed, reducing the dataset from **259,415 → 259,411 rows**:

| # | Station Name | MAC Address | Port | Start Date | End Date | Energy (kWh) | Plug In Event Id |
| :-: | :--- | :--- | :-: | :--- | :--- | --: | --: |
| 1 | `PALO ALTO CA / MPL #6` | `000D6F00015B7509` | 1 | `2/25/2015 15:24` | `2/25/2015 15:46` | 1.186864 | 469 |
| 2 | `PALO ALTO CA / HIGH #2` | `0024:B100:0002:5959` | 1 | `2/27/2017 11:14` | `2/27/2017 14:17` | 8.538000 | 1488222853 |
| 3 | `PALO ALTO CA / BRYANT #2` | `000D:6F00:009D:CFC3` | 2 | `3/10/2017 14:07` | `3/10/2017 15:53` | 4.385000 | 9453 |
| 4 | `PALO ALTO CA / BRYANT #5` | `0024:B100:0002:6D51` | 2 | `4/29/2019 8:10` | `4/29/2019 9:09` | 5.780000 | 3413 |

- **Why `Plug In Event Id` duplicates were NOT removed:** `Plug In Event Id` is a per-station sequential counter that starts from low integers on each hardware unit and resets after firmware reboots (222,577 repeated values across distinct stations/dates). Dropping on `Plug In Event Id` would erroneously destroy valid charging sessions.
- **Post-cleaning verification:** `0` exact duplicate rows remain across both the original 33 columns and all 78 cleaned columns.

---

## 4. Missing Values Analysis & Treatment

No rows were dropped due to missing optional metadata (`County`, `EVSE ID`, `System S/N`, `Model Number`, `Driver Postal Code`, `User ID`, `Currency`, `Ended By`, or `Transaction Date`).

### Original 33 Columns — Before vs. After Missing Counts

| Column | Raw Missing (n=259,415) | Cleaned Missing (n=259,411) | Treatment / Reason |
| :--- | --: | --: | :--- |
| `County` | 84,665 (32.64%) | 84,664 (32.64%) | Retained as null (−1 from duplicate pair #1); optional metadata, also contains known `San Mateo County` mislabeling on Palo Alto sites |
| `EVSE ID` | 78,948 (30.43%) | 78,947 (30.43%) | Retained as null (−1 from duplicate pair #1); structurally absent pre-mid-2016 |
| `System S/N` | 78,948 (30.43%) | 78,947 (30.43%) | Retained as null; co-missing with `EVSE ID` pre-mid-2016 |
| `Model Number` | 78,948 (30.43%) | 78,947 (30.43%) | Retained as null; co-missing with `EVSE ID` pre-mid-2016 |
| `Driver Postal Code` | 8,402 (3.24%) | 8,402 (3.24%) | Retained as null; unregistered/guest/international drivers |
| `User ID` | 7,677 (2.96%) | 7,677 (2.96%) | Retained as null; guest/anonymous sessions (plus 453 `"0"` values preserved) |
| `Currency` | 1,788 (0.69%) | 1,788 (0.69%) | Retained as null; mostly legacy free (`Fee = 0`) sessions |
| `Ended By` | 248 (0.10%) | 248 (0.10%) | Retained as null; frequently coincides with station connectivity/batch-upload events |
| `Transaction Date (Pacific Time)` | 209 (0.08%) | 209 (0.08%) | Retained as null in both raw column and `transaction_datetime_pt` |
| `Port Type` → **`port_type_clean`** | **9** (0.0035%) | Raw: 9 / **Clean: 0** | **Unambiguously recovered** in `port_type_clean` via 1-to-1 deterministic `Plug Type` mapping (`J1772` → `Level 2` [7 rows]; `NEMA 5-20R` → `Level 1` [2 rows]); tracked via `flag_port_type_recovered` |
| *All other 23 original columns* | **0** | **0** | Complete across all 259,411 records |

### New Derived Columns — Intentional Nulls Only

Out of the 45 newly added columns, **42 have 0 missing values**. Only 3 derived columns contain intentional nulls by design:
- `transaction_datetime_pt`: **209** nulls (exact match to genuine missing `Transaction Date (Pacific Time)` values — all 40 Excel serial rows were recovered).
- `idle_duration_min`: **45** nulls (intentionally set to `NaN` where `Charging Time > Total Duration` to prevent invalid negative idle durations).
- `implied_avg_power_kw`: **1,002** nulls (intentionally set to `NaN` where `charging_duration_sec < 60` because sub-minute denominators produce noisy power division artifacts).

---

## 5. Timestamp Corrections & Batch-Upload Mismatch Investigation

All timestamps were converted into consistent **Pacific Local Time** (`start_datetime_pt`, `end_datetime_pt`, `transaction_datetime_pt`) and **UTC** (`start_datetime_utc`, `end_datetime_utc`) using exact US Pacific Daylight Saving Time boundaries (2011–2021), while leaving the original raw timestamp columns untouched.

### 5.1 Excel Serial Date Correction (40 Records)
- **Finding:** Exactly **40 rows** had `End Date` and `Transaction Date (Pacific Time)` exported as Excel serial day floats (`1899-12-30` epoch): **24 rows** on `2017-04-20` (e.g., `42845.81041` → `2017-04-20 19:27:00 PDT`) and **16 rows** on `2017-08-29`.
- **Verification:** Across all 40 rows, the converted timestamps match `Start Date + Total Duration` with a maximum absolute discrepancy of **51 seconds** (pure minute-truncation rounding on `Start Date`).
- **Action:** Converted cleanly in `end_datetime_pt`, `end_datetime_utc`, and `transaction_datetime_pt`; flagged via `flag_excel_serial_date_fixed = True` (40 rows).

### 5.2 UTC-Encoded Timestamp Correction (32 Records)
- **Finding:** **31 rows** had `Start Time Zone = UTC` (causing all 31 naive `End Date < Start Date` errors observed in Phase 1), and **1 row** (`PALO ALTO CA / CAMBRIDGE #1`, `7/7/2017 13:11 PDT` → `7/8/2017 0:15 UTC`) had `End Time Zone = UTC`.
- **Verification:** Converting `UTC → Pacific Time` resolved **100%** of negative naive intervals (`end_before_start_utc_count = 0`). 31 of 32 records match `Total Duration` within ≤ 120 seconds; the remaining 1 row (`10/29/2013`, `HAMILTON #1`) had an additional ~3-hour batch offset caught by the `End Date` mismatch check below.
- **Action:** Converted accurately in `start_datetime_pt` / `end_datetime_pt` and flagged via `flag_start_utc_converted` (31 rows) and `flag_end_utc_converted` (1 row).

### 5.3 Investigation of `End Date` Batch-Upload Mismatches
After resolving the 40 Excel serials and 32 UTC rows, comparing `(end_datetime_utc − start_datetime_utc)` against `Total Duration (hh:mm:ss)` revealed:
- **318 records** with `|End_UTC − Start_UTC − Total_Duration| > 5 minutes` (`flag_end_date_mismatch_gt_5min = True`), of which **235 records** differ by `> 1 hour` (`flag_end_date_mismatch_gt_1h = True`):
  - **304 records** have `End Date` recorded *later* than `Start Date + Total Duration` (batch-upload lag after stations temporarily lost connectivity — 140 of the >1h cases have missing `Ended By`, such as `PALO ALTO CA / MPL #4` in May–June 2015 where multiple sessions from different days received identical upload `End Date`s).
  - **14 records** have `End Date` recorded *earlier* than `Start Date + Total Duration`, including **12 records** where `End Date == Start Date` (e.g., 7 consecutive sessions on `10/3/2013` at `BRYANT #2` with valid `Total Duration` and `Energy` but `End Date` copied from `Start Date`).
- **Critical Exception Discovered (40 Rollover Rows):** Of the 318 mismatched rows, **40 rows** are legacy 2011–2013 sessions where `Total Duration` **rolled over modulo 24 hours** (see Section 6.1). On those 40 rows, `End Date` reflects the true multi-day span while `Total Duration` is truncated, so `Start + Total Duration` must **not** be used to override `End Date`.
- **Action Taken:** For the **280 records** where `End Date` is mismatched (`> 5 min` or `End ≤ Start`) AND `Total Duration` is reliable (`not flag_duration_rollover_gt_60s`), `occupancy_end_utc` / `occupancy_end_pt` was derived as `start_datetime_utc + total_duration_sec`, original `End Date` was retained untouched, and the correction was logged via `flag_occupancy_interval_corrected = True` and `occupancy_end_source = "derived_start_plus_total_duration"`.

---

## 6. Duration and Energy Anomalies

All sessions were **retained** (0 sessions deleted for duration or power anomalies) and transparently flagged.

### 6.1 Forensic Investigation of `Charging Time > Total Duration` (45 Records)
Inspection of all 45 rows revealed **two distinct root causes**:
1. **Modulo-24-Hour `Total Duration` Rollover (40 rows, `flag_duration_rollover_gt_60s = True`):** Legacy 2011–2013 sessions lasting >24 hours where `Total Duration (hh:mm:ss)` wrapped around modulo 24h (e.g., `1/10/2013 09:51 → 1/11/2013 10:23` = 24h 32m elapsed, logged as `Total Duration = 0:31:03` while `Charging Time = 5:33:11`).
2. **Minor Sub-Minute Clock Sync Differences (5 rows):** `Charging Time` exceeded `Total Duration` by only `1–12 seconds` (e.g., `4:05:56` vs. `4:05:55`).

**Treatment:**
- Flagged all 45 rows with `flag_charging_gt_total_duration = True` and `flag_duration_unreliable = True`.
- Set `idle_duration_min = NaN` for all 45 rows (`0` negative idle times remain in the cleaned dataset).
- Preserved original `End Date` for the 40 rollover rows and flagged `flag_occupancy_interval_unreliable = True`.

### 6.2 Extreme Durations & Implausible Power Values
| Anomaly Category | Flag Column | Flagged Count | Treatment |
| :--- | :--- | --: | :--- |
| `Total Duration > 24 hours` | `flag_extreme_total_duration_gt_24h` | **22** | Retained & flagged (includes 6 reboot-stuck sessions `> 48h`, max `114.5h`, also marked `flag_occupancy_interval_unreliable = True`) |
| `Charging Time > 12 hours` | `flag_extreme_charging_time_gt_12h` | **213** | Retained & flagged (primarily legitimate long Level 1 & overnight Level 2 sessions; max `22.9h`) |
| `Implied Power > 20 kW` (all sessions) | `flag_implausible_power_gt_20kw` | **10** *(9 with chg ≥ 1m)* | Retained & flagged; valid session demand/energy records are NOT removed merely because implied power telemetry was unusual |
| Negative / Zero `Energy (kWh)` | — | **0** | Verified `min = 0.01 kWh`, `max = 97.36 kWh`, `0` missing |

---

## 7. Identifier Standardization

| Identifier | Raw Column (Preserved) | Standardized Column | Before → After | Evidence & Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Station Name** | `Station Name` | `station_id` | **47 → 46** unique | Fixed **13 rows** of `"PALO ALTO CA / BRYANT # 1"` → `"PALO ALTO CA / BRYANT #1"`, verified by 100% matching MAC (`0024B1000003CF38`), EVSE ID (`5358941`), address (`445 Bryant St`), and model (`CT4010-HD-GW`). No other stations were merged. |
| **MAC Address** | `MAC Address` | `mac_address_clean` | **83 → 68** unique physical units | Stripped colons and uppercased (e.g., `000D:6F00:015A:9D76` → `000D6F00015A9D76`; 222,064 rows had colons). Collapses the 15 dual-format MACs into single physical unit IDs while keeping raw `MAC Address` intact. |
| **Org Name** | `Org Name` | `org_name_clean` | 2 raw strings → **1** (`City of Palo Alto`) | Stripped trailing whitespace on **236,861 rows**. |

---

## 8. Prepared Demand Features & Validated Occupancy Intervals

### 8.1 Basic Demand-Prediction Fields (No Target or Lag Leakage)
Prepared in `dataset/processed/ev_charging_cleaned.csv` for downstream Phase 4 Feature Engineering:
- **Temporal (Pacific local time):** `start_date` (`YYYY-MM-DD`), `hour_of_day` (`0–23`), `day_of_week` (`0=Mon..6=Sun`), `day_of_week_name`, `month` (`1–12`), `year` (`2011–2020`), `is_weekend` (`True`/`False`).
- **Entity & Demand Metrics:** `station_id`, `mac_address_clean`, `port_type_clean`, `energy_kwh`, `charging_duration_sec` / `charging_duration_min`, `total_duration_sec` / `total_duration_min`, `idle_duration_min`, `implied_avg_power_kw`.

### 8.2 Physical Occupancy vs. Active Charging Intervals (No Assumed Capacity)
- **Physical Bay Occupancy Interval:** `[occupancy_start_pt, occupancy_end_pt]` (and DST-safe `[occupancy_start_utc, occupancy_end_utc]`), built using validated `Total Duration` whenever `End Date` suffered batch-upload drift.
- **Active Charging Interval:** `[occupancy_start_pt, active_charging_end_pt]` (and UTC equivalent), built from `Start + Charging Time` to separate active electrical load from idle overstay.
- **Audit Breakdown of `occupancy_end_source`:**
  - `original_end_date`: **259,050** (99.861%)
  - `derived_start_plus_total_duration`: **280** (0.108%) — repaired batch-upload & zero-span `End Date` rows
  - `original_end_date_duration_rollover`: **40** (0.015%) — retained multi-day `End Date` on modulo-24h duration rollover rows (flagged unreliable)
  - `excel_serial_converted`: **40** (0.015%) — recovered Excel serial dates
  - `utc_converted`: **1** (0.0004%) — recovered UTC `End Date`

### Occupancy Validation Results (Before vs. After Interval Correction)

| Concurrency Sanity Metric | Before (`end_datetime_utc`) | After (`occupancy_end_utc`) | Impact |
| :--- | --: | --: | :--- |
| Same-`(MAC, Port)` all overlaps | 299 | **69** | **−76.9%** |
| Same-`(MAC, Port)` overlaps `> 2 min` | 251 | **58** | **−76.9%** |
| Same-`(MAC, Port)` overlaps `> 10 min` | 234 | **55** | **−76.5%** |
| `PALO ALTO CA / MPL #4` max simultaneous sessions | 7 *(artificial batch-upload stack)* | **2** | Resolved artificial 7-way stack |
| City-wide max simultaneous sessions | 45 | **45** | Consistent |
| Non-positive occupancy intervals (`end ≤ start`) | 12 | **0** | **100% resolved** |

> **Capacity Disclaimer:** Per project rules, no station port capacities or occupancy utilization percentages (`active / capacity`) were invented or added.

---

## 9. Post-Cleaning Quality Checks

All 9 automated assertions in `scripts/phase3_clean_data.py` passed:

| Check | Expected | Actual | Status |
| :--- | :--- | :--- | :-: |
| Original raw CSV unchanged (SHA-256) | Exact match | `e65f5f5d3861cdd6...` | **PASS** |
| Exact full-row duplicates remaining | 0 | **0** | **PASS** |
| Unparseable `start_datetime_pt` or `end_datetime_pt` | 0 | **0** | **PASS** |
| Negative or zero occupancy intervals | 0 | **0** | **PASS** |
| Negative `idle_duration_min` values | 0 | **0** | **PASS** |
| Missing `port_type_clean` values | 0 | **0** | **PASS** |
| Missing or non-positive `energy_kwh` values | 0 | **0** | **PASS** |
| Standardized station count (`station_id`) | 46 | **46** | **PASS** |
| Standardized physical MAC count (`mac_address_clean`) | 68 | **68** | **PASS** |

---

## 10. Retained Anomalies & Unresolved Issues (Documented for Phase 4)

1. **55 Residual Same-`(MAC, Port)` Overlaps (>10 min):** Mostly stem from the **40 legacy modulo-24h duration rollover rows** and **6 reboot-stuck `>48h` rows** (all 46 explicitly flagged via `flag_occupancy_interval_unreliable = True`), plus ~9 minor legacy telemetry overlaps. Downstream occupancy aggregations can filter `~flag_occupancy_interval_unreliable` for clean concurrency counts without dropping those sessions from total arrival/energy counts.
2. **Legacy Hardware Metadata Sparsity:** `EVSE ID`, `System S/N`, and `Model Number` remain structurally missing before mid-2016 (78,947 rows), and 53.68% of non-null `System S/N` values were truncated to scientific notation by an upstream Excel export prior to dataset publication.
3. **Address & County Text Variations:** Raw `Address 1` still contains 20 string variants across ~8 physical parking hubs (e.g., `528 High St` / `528 High St.` / `528 high st`), and `County` labels 27,487 Palo Alto sessions as `San Mateo County`. Hub/location grouping can be cleanly engineered in Phase 4 using `station_id` and coordinates `(Latitude, Longitude)` without altering raw geographic fields.
4. **Foreign Billing Currencies (14 rows) & Non-Numeric/Zero `User ID`s:** Retained untouched; `Fee` reflects roaming driver account currency on 4 paid foreign-currency rows (`EUR`, `MXN`).
5. **Structural Regime Shifts (August 2017 Fee Rollout & 2020 COVID-19):** Intentionally preserved in full; regime indicators and chronological train/validation/test splitting should be designed explicitly in Phase 4 (Feature Engineering) and Phase 5 (Modeling).
