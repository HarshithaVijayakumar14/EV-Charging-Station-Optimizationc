# Phase 4: Feature Engineering Report

**Project:** Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations  
**Input Dataset (Unmodified):** `dataset/processed/ev_charging_cleaned.csv` (259,411 sessions × 78 columns)  
**Output Feature Dataset:** `dataset/processed/ev_charging_hourly_features.csv` (**1,533,050 rows × 44 columns**, 427.34 MB)  
**Feature Engineering Script:** `scripts/phase4_feature_engineering.py`

> **Integrity Guarantee:** Neither the original raw CSV (`EVChargingStationUsage.csv`) nor the Phase 3 cleaned dataset (`dataset/processed/ev_charging_cleaned.csv`) was modified; SHA-256 hashes were verified identical before and after execution. No ML models were trained, no random shuffling was applied, and no station capacities or utilization percentages were invented.

---

## 1. Prediction Targets & Exact Semantics

### Primary Target: `sessions_count`
- **Definition:** The integer number of EV charging sessions **arriving** (plugging in) at a specific `station_id` during a given Pacific Time clock hour `[hour_timestamp_pt, hour_timestamp_pt + 1 hour)`.
- **Distinction from Occupancy:** `sessions_count` measures **new hourly arrival flow** (how many drivers initiate charging during that hour), **not** the stock of vehicles already sitting plugged in at the station.
- **Distribution Across 1,533,050 Station-Hours:**
  - **Total arrivals conserved:** **259,411** (100% of cleaned Phase 3 sessions retained, including arrivals whose occupancy intervals were flagged unreliable).
  - **Mean / Std Dev:** `0.1692` / `0.4170` sessions per station-hour; **Max:** `5` arrivals in a single station-hour.
  - **Zero-arrival hours:** `1,298,074` (**84.67%**) — expected for hourly station-level data including overnight and quiet periods.
  - **Exact value counts:** `0`: 1,298,074 | `1`: 211,559 | `2`: 22,446 | `3`: 928 | `4`: 39 | `5`: 4.

### Optional Secondary Target: `energy_kwh_arrival_sum`
- **Definition:** Total `energy_kwh` across all charging sessions whose arrival falls in `(station_id, hour_timestamp_pt)` (`0.0` when `sessions_count == 0`).
- **Total energy conserved:** **2,216,493.71 kWh** (exact match to Phase 3 cleaned total); Mean: `1.4458 kWh`/station-hour; Max: `97.36 kWh`.
- **Important Semantic Caveat:** Assigning a session's entire `energy_kwh` to its **arrival hour** represents *energy demand committed by arrivals in that hour*, which is **not** identical to the electrical energy metered inside that specific clock hour (because a session starting at `11:45` and charging for 3 hours draws power across `11:00`, `12:00`, `13:00`, and `14:00`). **`sessions_count` remains the primary ML prediction objective.**

---

## 2. Handling Missing Station-Hour Periods & DST Consistency

Building a valid time-series panel for 46 stations that launched across 9.4 years requires solving two subtle pitfalls:

1. **No False Pre-Commissioning or Post-Shutdown Zeros:**
   - Rather than creating a naive `46 stations × 82,630 hours` Cartesian grid (~3.80M rows, which would fabricate >1.1M false zeros before stations like `SHERMAN 1–17` or `CAMBRIDGE #1–5` even existed), each station's hourly grid spans strictly from its **own first recorded arrival hour (`first_hour_utc`) to its own last recorded arrival hour (`last_hour_utc`)**.
   - This also avoids inventing ~34,865 post-shutdown zeros after the 11 stations that stopped reporting prior to December 2020 (e.g., `RINCONADA LIB 1–3`, `BRYANT #2–3`, `HAMILTON #1–2`, `MPL #1–3`, `TED THOMPSON #1`).
   - Within each station's active operating span `[first_hour_utc, last_hour_utc]`, **every single clock hour is preserved contiguously** (filling hours with no arrivals as `sessions_count = 0` and `energy_kwh_arrival_sum = 0.0`) so that `shift(1)`, `shift(24)`, and `shift(168)` always correspond to **exact physical clock hours** (1h, 24h, and 168h ago).
2. **Exact Daylight Saving Time (DST) Spine Construction:**
   - Each station's contiguous hourly spine is constructed in **UTC** (`hour_timestamp_utc`, verified exact 3,600-second steps with zero gaps or duplicates) and mapped to offset-aware Pacific Time (`hour_timestamp_pt`, e.g., `2015-11-01 01:00:00-07:00` vs. `2015-11-01 01:00:00-08:00`).
   - This eliminates Fall-Back `01:00` duplicate collisions and Spring-Forward phantom `02:00` hours while keeping all calendar features (`hour`, `day`, `day_of_week`, `month`, `year`) in true Pacific local wall-clock time.

---

## 3. Complete Feature Dictionary (All 44 Columns)

### A. Identifiers & Targets (5 Columns)

| Column | Type | Missing | Definition |
| :--- | :--- | --: | :--- |
| `station_id` | string | 0 | Standardized station identifier (46 unique stations from Phase 3) |
| `hour_timestamp_pt` | ISO-8601 | 0 | Pacific local clock hour with explicit `-07:00` (PDT) / `-08:00` (PST) offset |
| `hour_timestamp_utc` | ISO-8601 | 0 | Contiguous UTC clock hour (`YYYY-MM-DD HH:00:00`); primary chronological sort key |
| `sessions_count` | int32 | 0 | **Primary target:** number of charging sessions arriving at `station_id` during this hour |
| `energy_kwh_arrival_sum` | float64 | 0 | **Secondary target:** sum of `energy_kwh` for sessions arriving during this hour |

### B. Time-Based, Calendar & Regime Features (15 Columns — Known in Advance, Zero Leakage)

| Column | Type | Missing | Definition |
| :--- | :--- | --: | :--- |
| `hour` | int8 | 0 | Pacific local hour of day (`0–23`) |
| `day` | int8 | 0 | Pacific local day of month (`1–31`) |
| `day_of_week` | int8 | 0 | Pacific local weekday (`0` = Monday … `6` = Sunday) |
| `month` | int8 | 0 | Pacific local calendar month (`1–12`) |
| `year` | int16 | 0 | Pacific local calendar year (`2011–2020`) |
| `is_weekend` | int8 | 0 | `1` if Saturday or Sunday, else `0` |
| `is_us_federal_holiday` | int8 | 0 | `1` if Pacific calendar date is an observed US federal holiday (deterministic calendar rule), else `0` |
| `is_working_day` | int8 | 0 | `1` if Monday–Friday AND not a US federal holiday, else `0` |
| `hour_sin`, `hour_cos` | float64 | 0 | Cyclical encoding of hour: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$ |
| `dow_sin`, `dow_cos` | float64 | 0 | Cyclical encoding of day of week: $\sin(2\pi \cdot \text{dow}/7)$, $\cos(2\pi \cdot \text{dow}/7)$ |
| `month_sin`, `month_cos` | float64 | 0 | Cyclical encoding of month: $\sin(2\pi \cdot (\text{month}-1)/12)$, $\cos(2\pi \cdot (\text{month}-1)/12)$ |
| `is_paid_era` | int8 | 0 | Policy pricing regime indicator: `1` from `2017-08-01` onward (observed 0% → 95–99% paid fee shift in Phase 2), else `0` |

### C. Station-Level Strictly Causal Historical Features (8 Columns — Past Hours `≤ t−1` Only)

| Column | Type | Missing | Definition |
| :--- | :--- | --: | :--- |
| `station_age_days` | float64 | 0 | Days elapsed since the station's first recorded arrival hour (`(t − first_hour_utc) / 24h`) |
| `station_cum_sessions_prior` | int64 | 0 | Cumulative arrivals at `station_id` strictly prior to hour `t` (`0` at hour 0) |
| `station_expanding_mean_sessions` | float64 | 46 | Expanding average hourly arrivals at `station_id` over `[0 .. t−1]` (`NaN` at each station's first hour) |
| `station_hour_expanding_mean_sessions` | float64 | 989 | Expanding average arrivals at `(station_id, hour)` across strictly prior days at that exact hour of day |
| `hours_since_last_session` | float64 | 46 | Clock hours elapsed since the most recent non-zero arrival at `station_id` in `[0 .. t−1]` (helps capture quiet/idle stretches without assuming downtime) |
| `station_hist_mean_energy_per_session` | float64 | 46 | Causal expanding mean `energy_kwh` per session at `station_id` over arrivals strictly prior to hour `t` |
| `station_hist_mean_plugged_dur_min` | float64 | 46 | Causal expanding mean **physical plugged-in duration** (`total_duration_min`) strictly prior to `t`, **excluding** the 67 `flag_duration_unreliable` rows |
| `station_hist_mean_active_chg_min` | float64 | 46 | Causal expanding mean **active charging duration** (`charging_duration_min`) strictly prior to `t`, **excluding** the 67 `flag_duration_unreliable` rows |

### D. Causal Pre-Hour Empirical Concurrency Features (2 Columns — Excludes Unreliable Intervals, No Capacity Invented)

| Column | Type | Missing | Definition |
| :--- | :--- | --: | :--- |
| `prior_plugged_in_vehicles_at_hour_start` | int16 | 0 | Count of valid sessions at `station_id` with `occupancy_start_utc < t < occupancy_end_utc` (vehicles **already physically plugged in** strictly before hour `t` begins; excludes the 46 `flag_occupancy_interval_unreliable` sessions; mean `0.4177`, max `6`) |
| `prior_active_charging_vehicles_at_hour_start` | int16 | 0 | Count of valid sessions at `station_id` with `occupancy_start_utc < t < active_charging_end_utc` (vehicles **already actively charging** strictly before hour `t` begins; excludes unreliable occupancy & duration sessions; mean `0.3366`, max `6`) |

### E. Leakage-Safe Historical Demand Lag & Rolling Features (14 Columns — Strictly Shifted Before Aggregation)

| Column | Type | Missing *(Warm-Up)* | Formula (Per `station_id` on Contiguous UTC Spine) |
| :--- | :--- | --: | :--- |
| `lag_1h_sessions` | float64 | 46 | `sessions_count[t − 1]` |
| `lag_2h_sessions` | float64 | 87 | `sessions_count[t − 2]` |
| `lag_3h_sessions` | float64 | 128 | `sessions_count[t − 3]` |
| `lag_24h_sessions` | float64 | 989 | `sessions_count[t − 24]` (same hour yesterday) |
| `lag_168h_sessions` | float64 | 6,804 | `sessions_count[t − 168]` (same hour 7 days ago) |
| `rolling_mean_24h_sessions` | float64 | 989 | Mean of `sessions_count[t − 24 .. t − 1]` (`min_periods=24`) |
| `rolling_sum_24h_sessions` | float64 | 989 | Sum of `sessions_count[t − 24 .. t − 1]` (`min_periods=24`) |
| `rolling_mean_168h_sessions` | float64 | 6,804 | Mean of `sessions_count[t − 168 .. t − 1]` (`min_periods=168`) |
| `rolling_sum_168h_sessions` | float64 | 6,804 | Sum of `sessions_count[t − 168 .. t − 1]` (`min_periods=168`) |
| `lag_1h_energy_kwh` | float64 | 46 | `energy_kwh_arrival_sum[t − 1]` |
| `lag_24h_energy_kwh` | float64 | 989 | `energy_kwh_arrival_sum[t − 24]` |
| `lag_168h_energy_kwh` | float64 | 6,804 | `energy_kwh_arrival_sum[t − 168]` |
| `rolling_mean_24h_energy_kwh` | float64 | 989 | Mean of `energy_kwh_arrival_sum[t − 24 .. t − 1]` (`min_periods=24`) |
| `rolling_mean_168h_energy_kwh` | float64 | 6,804 | Mean of `energy_kwh_arrival_sum[t − 168 .. t − 1]` (`min_periods=168`) |

---

## 4. How Data Leakage Was Prevented & Mathematically Proven

1. **Shift-Before-Aggregate Rule:** Every rolling (`24h`, `168h`) and expanding (`station_expanding_mean_*`, `station_hour_expanding_mean_*`, `station_cum_sessions_prior`, `station_hist_mean_*`, `hours_since_last_session`) feature is computed strictly on `series.shift(1)` per `station_id`. At timestamp `t`, the window covers `[t − W .. t − 1]` (or `[0 .. t − 1]`) and **never** touches `t` or `t + k`.
2. **Strict Pre-Hour Occupancy Boundary:** `prior_plugged_in_vehicles_at_hour_start` and `prior_active_charging_vehicles_at_hour_start` enforce strict inequality `start < t < end`, counting only vehicles already connected at real-time forecast origin `t:00:00` before any arrivals in hour `t` occur.
3. **Adversarial Perturbation Proof:** Built directly into `scripts/phase4_feature_engineering.py`, we mutated `sessions_count`, `energy_kwh_arrival_sum`, and duration aggregates on `PALO ALTO CA / HAMILTON #2` by `+9999` at cutoff `t* = 1000` and all future hours `t >= t*`, recomputed all engineered features, and verified that **100% of engineered feature columns at every timestamp `t <= t*` (including `t = t*` itself) remained bit-for-bit identical**.

---

## 5. Dataset Dimensions, Yearly Breakdown & Chronological ML Readiness

The output CSV is **sorted strictly chronologically by `(hour_timestamp_utc, station_id)`** (never randomly shuffled). Out of `1,533,050` total station-hours, **1,526,246 rows (99.56%)** have complete 168-hour lag histories; the remaining **6,804 rows (0.44%)** are initial per-station warm-up hours (first 168 hours of each station, plus 7 late-Dec-2020 `SHERMAN` stations with <168 hours of total span).

| Year | Station-Hours | Active Stations | Total Sessions | Total Energy (kWh) | Zero-Arrival Hours (%) |
| :-: | --: | --: | --: | --: | --: |
| 2011 | 14,134 | 5 | 1,275 | 8,368.06 | 91.12% |
| 2012 | 43,920 | 5 | 6,601 | 42,974.12 | 85.47% |
| 2013 | 43,800 | 5 | 10,109 | 71,901.40 | 78.27% |
| 2014 | 57,572 | 11 | 11,722 | 88,389.09 | 80.97% |
| 2015 | 133,724 | 17 | 23,110 | 181,464.16 | 83.86% |
| 2016 | 191,630 | 22 | 46,943 | 405,569.12 | 78.16% |
| 2017 | 222,888 | 27 | 49,020 | 422,242.36 | 80.45% |
| 2018 | 277,536 | 33 | 42,803 | 360,270.74 | 86.06% |
| 2019 | 289,080 | 33 | 47,731 | 437,690.28 | 85.25% |
| 2020 | 258,766 | 46 | 20,097 | 197,624.38 | 92.85% |
| **Total** | **1,533,050** | **46** | **259,411** | **2,216,493.71** | **84.67%** |

- **Guidance for Phase 5 Chronological Splitting (not executed here):** Because 2020 experienced the COVID-19 demand collapse (−71% Mar–Dec vs. 2019), Phase 5 should evaluate models both on a **pre-COVID chronological benchmark** (e.g., Train: `2011–2018` [or paid-era focus], Validation: `2019 H1`, Test: `2019 H2`) and on **2020 out-of-distribution stress testing**, fitting any encoders/scalers strictly on the training split only.

---

## 6. Automated Quality & Anti-Leakage Checks (19 / 19 PASS)

| Check Name | Result | Verified Metric |
| :--- | :-: | :--- |
| `zero_missing_primary_target` | **PASS** | 0 missing values in `sessions_count` |
| `zero_missing_secondary_target` | **PASS** | 0 missing values in `energy_kwh_arrival_sum` |
| `exact_session_arrival_conservation` | **PASS** | `sum(sessions_count) == 259,411` |
| `exact_energy_conservation` | **PASS** | `sum(energy_kwh_arrival_sum) == 2,216,493.71` kWh |
| `zero_negative_sessions_count` | **PASS** | Min `sessions_count = 0` |
| `zero_negative_energy` | **PASS** | Min `energy_kwh_arrival_sum = 0.0` |
| `zero_negative_prior_occupancy` | **PASS** | Min `prior_plugged = 0`, Min `prior_active = 0` |
| `zero_duplicate_station_hour_utc` | **PASS** | 0 duplicate `(station_id, hour_timestamp_utc)` rows |
| `zero_duplicate_station_hour_pt` | **PASS** | 0 duplicate `(station_id, hour_timestamp_pt)` rows |
| `exact_1h_contiguity_all_stations` | **PASS** | Every consecutive row per station differs by exactly 3,600s |
| `zero_invalid_timestamps` | **PASS** | 0 null or unparseable UTC/Pacific timestamps |
| `exact_lag_1h/2h/3h/24h/168h_sessions` (5 checks) | **PASS** | 0 value or NaN-position mismatches across all 1,533,050 rows |
| `exact_rolling_24h_verified` | **PASS** | Independent manual slice checks match within `< 1e-5` |
| `exact_rolling_168h_verified` | **PASS** | Independent manual slice checks match within `< 1e-5` |
| `zero_future_leakage_perturbation_proof` | **PASS** | +9,999 target mutation at `t >= t*` produced **0** feature changes for `t <= t*` |

---

## 7. Remaining Limitations

1. **Zero-Inflated Count Target (84.67% Zeros):** Standard RMSE/MAE optimization on highly zero-inflated hourly counts can under-predict rare peak bursts (`sessions_count >= 2`); Phase 5 should consider count-appropriate objectives (e.g., Poisson / Tweedie loss in gradient boosting) alongside RMSE and MAE.
2. **Unobserved Short-Term Downtime Within Active Spans:** While pre-commissioning and post-shutdown false zeros were avoided, internal multi-day zero stretches during temporary hardware outages remain indistinguishable from genuine zero demand (partially mitigated by the causal `hours_since_last_session` feature).
3. **Forecast-Horizon Availability of Real-Time Features:** `lag_1h/2h/3h_sessions` and `prior_plugged_in_vehicles_at_hour_start` are directly observable for **1-hour-ahead** real-time forecasting. For **24-hour-ahead** day-ahead planning, Phase 5 should either train a dedicated day-ahead feature subset (`lag_24h`, `lag_168h`, calendar, station expanding means) or forecast recursively.
4. **Cold-Start Late-Opening Stations:** 7 `SHERMAN` stations opened in late December 2020 with `< 168 hours` total span (5 of them have only 1 recorded hour), so their `lag_168h` / `rolling_168h` features remain naturally `NaN` rather than fabricated.
5. **Residual Legacy Telemetry Overlaps:** Excluding the 46 Phase 3 `flag_occupancy_interval_unreliable` rows removed most batch-upload stacks, though ~9 unflagged legacy overlapping sessions remain, resulting in a maximum observed `prior_plugged_in_vehicles_at_hour_start` of `6`.
6. **External Exogenous Variables Absent:** Weather, temperature, Stanford/Palo Alto local events, and parking pricing changes are not included in the source CSV.
