# Electric Vehicle Charging Station Optimization: Predicting Demand for EV Stations
## Final Academic Project Report

**Degree / Course:** Capstone Project in Machine Learning & Software Engineering  
**Academic Year:** 2026–2027  
**Project Title:** Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations  
**Case Study Location:** City of Palo Alto, California, USA  
**Dataset Coverage:** July 2011 – December 2020 (10 Years; 259,411 Cleaned Sessions; 46 Stations)  
**System Architecture:** Flask REST API Backend + React 18 / Vite / Tailwind CSS Frontend + XGBoost Regressors  
**Report Document:** Final Submission Project Report  
**Date:** November 2026  

---

## 1. Abstract

The worldwide transition toward electric mobility requires robust, data-driven municipal infrastructure management. While physical charging networks expand rapidly, charging demand exhibits extreme temporal and spatial volatility, resulting in severe commuter-peak bottlenecks, grid strain, and inefficient resource allocation. 

This project designs, implements, and validates an end-to-end Machine Learning and Decision Support System for municipal electric vehicle charging demand prediction and planning optimization. Using 10 years of empirical charging telemetry from the City of Palo Alto, California, we establish a rigorous data cleaning pipeline (resolving timestamp corruption, timezone offsets, and deduplication) and construct a contiguous hourly panel of 1,533,050 station-hours across 46 public charging stations. 

We formalize the core predictive task as forecasting hourly session arrival flows (`sessions_count`) under strict chronological data splits to eliminate data leakage: training on 2011–2017 (707,668 station-hours), tuning on 2018 validation data (277,536 station-hours), testing on untouched 2019 data (289,080 station-hours), and stress-testing under the COVID-19 pandemic shock in 2020 (258,766 station-hours). XGBoost Regressors were trained for two operational horizons: a real-time 1-hour-ahead forecast ($T_{\text{last}} + 1\text{h}$) and a strictly causal day-ahead 24-hour forecast ($\le t - 24\text{h}$). 

On the held-out 2019 test set, XGBoost achieved superior predictive accuracy over baseline methods, recording a Root Mean Squared Error (RMSE) of 0.3564, a Mean Absolute Error (MAE) of 0.2029, and a volume calibration bias of only -3.43%. In the 2020 stress test, Random Forest exhibited slightly lower overall RMSE (0.2661 vs. 0.2696) due to heavy ensemble averaging during the pandemic collapse, while XGBoost proved markedly superior on newly commissioned cold-start stations ($R^2 = +0.242$ vs. $-0.197$). 

The models are deployed in a unified, production-grade web application featuring interactive telemetry analytics, dual-horizon demand forecasting, automated historical backtesting, cold-start safety guards, and explainable rule-based demand-planning suggestions.

---

## 2. Introduction

Electrification of the transport sector is a critical pillar of global carbon decarbonization. However, the operational efficiency of public charging infrastructure is severely challenged by the stochastic arrival behavior of EV drivers. Public chargers cannot be scheduled like private home chargers; their utilization is dictated by commute rhythms, retail operating hours, weather, and proximity to commercial centers.

Without predictive visibility, municipal operators face two compounding problems:
1. **Unanticipated Arrival Surges:** Morning commuter influxes cause station queues, driver frustration, and localized distribution grid demand spikes.
2. **Suboptimal Infrastructure Investment:** Municipal planners often lack objective metrics to differentiate between stations with chronic peak-hour demand and stations with consistently low utilization.

Machine learning offers a powerful paradigm for municipal energy management. By mining years of historical transaction logs, predictive algorithms can learn complex temporal patterns, seasonalities, and spatial characteristics to forecast arrival flows before they occur. This project delivers an integrated system that transforms raw municipal charging records into real-time operational intelligence.

---

## 3. Problem Statement

Given historical charging transaction records from a municipal EV network comprising $S$ public charging stations over time $t = 1, \dots, T$, the primary objective is to accurately predict the number of charging sessions arriving at each station in future time intervals without data leakage:

$$\hat{y}_{s, t+h} = f(\mathcal{X}_{s, \le t})$$

where:
- $\hat{y}_{s, t+h}$ denotes the predicted session arrival count at station $s$ at horizon $h \in \{1\text{ hour}, 24\text{ hours}\}$.
- $\mathcal{X}_{s, \le t}$ denotes the feature vector constructed strictly from historical observations available up to time $t$.

### Key Technical Challenges
1. **Count Sparsity & Zero Inflation:** Over 85% of all station-hours have zero arrivals ($y = 0$). Standard linear models collapse or over-predict false surges.
2. **Strict Causality & Anti-Leakage:** Multi-step day-ahead forecasting must not use future observations or recursive synthetic predictions that compound errors.
3. **Cold-Start Handling:** New stations commissioned late in the timeline lack historical training records. Models must handle these sites safely without hallucinated features.
4. **Data Boundary Discipline:** Municipal datasets record vehicle arrival timestamps and energy draw, but lack verified physical port counts, queue lengths, or downtime logs. The system must explicitly avoid ungrounded claims of physical charger congestion.

---

## 4. Project Objectives

1. **Phase 1 – Dataset Understanding:** Inspect data schemas, column types, missingness, and structural anomalies across all 259,415 raw records.
2. **Phase 2 – Exploratory Data Analysis (EDA):** Identify diurnal curves, commuter surges, weekend drops, Level 1 vs. Level 2 usage, and multi-year growth.
3. **Phase 3 – Data Cleaning & Preprocessing:** Resolve timestamp corruptions, standardize timezones to Pacific Time with DST offsets, recover missing port types, and remove exact duplicates while preserving raw data integrity.
4. **Phase 4 – Feature Engineering & Contiguous Panel Construction:** Formulate a contiguous hourly panel of 1,533,050 station-hours across 46 stations, engineering calendar harmonics, causal lags, rolling statistics, and concurrent vehicle metrics.
5. **Phase 5 – Model Training & Benchmarking:** Implement strict chronological splits (Train 2011–2017, Val 2018, Retrain 2011–2018, Test 2019, OOD 2020), comparing Tree models against linear and heuristic baselines.
6. **Phase 6 – Web Application Development:** Construct an enterprise-grade Flask + React single-page application integrating pre-trained models, interactive charts, and explainable rule-based planning recommendations.
7. **Phase 7 – Final Testing & Documentation:** Perform rigorous regression testing, verify model parity, investigate telemetry discrepancies, and compile complete academic documentation.

---

## 5. Existing System and Its Limitations

Traditional municipal EV network planning relies predominantly on static heuristics and retrospective reporting:
- **Static Historical Averages:** Municipalities evaluate charging stations using aggregate monthly or annual metrics (e.g., total kWh per month). These metrics fail to capture high-frequency diurnal surges.
- **Unadjusted Nameplate Capacity Assumptions:** Planners often assume nominal availability across all hours, ignoring commuter clustering and overnight lulls.
- **Randomized Train/Test Splitting in Literature:** Many published academic studies evaluate EV demand using randomized cross-validation. In time-series forecasting, randomly shuffling observations leaks future temporal structures into training sets, producing artificially inflated $R^2$ scores that fail in production.
- **Lack of Day-Ahead Causal Forecasting:** Existing systems either focus exclusively on 1-step-ahead forecasts or rely on recursive multi-step forecasting that compounds prediction errors across 24 hours.

---

## 6. Proposed System

The proposed system addresses these deficiencies through a multi-tier engineering architecture:
- **Contiguous Panel Modeling:** Models station-hours as a contiguous grid, explicitly capturing periods of zero demand rather than modeling only active transactions.
- **Dual-Horizon XGBoost Architecture:**
  - *1-Hour-Ahead Model:* High-frequency operational regressor incorporating immediate preceding arrival lags (`lag_1h`) and pre-hour concurrent vehicle counts.
  - *24-Hour Day-Ahead Model:* Purely causal planning regressor constructed strictly from features $\le t - 24\text{h}$, enabling day-ahead municipal scheduling without recursive error drift.
- **Cold-Start & Insufficient-History Safeguards:** Stations with fewer than 24 hours of telemetry are programmatically withheld from lag calculations with clear user notifications.
- **Explainable Recommendation Engine:** Translates raw numerical forecasts into transparent operational suggestions (e.g., highlighting peak concentration ratios) coupled with mandatory capacity limitation notices.
- **Unified Web Delivery:** Integrates the compiled React single-page application into Flask, allowing single-command deployment without complex microservice orchestration.

---

## 7. Dataset Description

The project utilizes the public municipal charging dataset provided by the City of Palo Alto, California, covering public chargers located at civic centers, libraries, parking structures, and parks.

### Dataset Dimensions & Attributes
- **Raw Dimensions:** 259,415 records $\times$ 33 attributes (85.45 MB CSV).
- **Temporal Span:** July 29, 2011 to December 31, 2020 (3,443 calendar days).
- **Core Raw Attributes:**
  - `Station Name`: Human-readable identifier (e.g., `PALO ALTO CA / HAMILTON #2`).
  - `MAC Address` / `EVSE ID`: Hardware network identifiers.
  - `Start Date` / `End Date`: Charging session connection timestamps.
  - `Total Duration (hh:mm:ss)`: Total plugged-in duration.
  - `Charging Time (hh:mm:ss)`: Active energy transfer duration.
  - `Energy (kWh)`: Net electrical energy delivered to the battery.
  - `Port Type`: Charging power tier (`Level 2` vs. `Level 1`).
  - `Latitude` / `Longitude`: Spatial coordinates in Palo Alto.

---

## 8. Data Preprocessing & Cleaning (Phase 3)

Data cleaning was executed without altering the original raw CSV, generating a verified cleaned dataset at `dataset/processed/ev_charging_cleaned.csv` (259,411 rows $\times$ 78 columns):

1. **Exact Deduplication:** Removed 4 confirmed exact full-row duplicate records (-19.89 kWh). Retained non-duplicate records that shared `Plug In Event Id` across distinct hardware ports.
2. **Timestamp Standardisation:**
   - Corrected 40 Excel serial date timestamps (e.g., numeric floats around `42370.0`).
   - Standardized all timestamps into Pacific Time (`America/Los_Angeles`), applying dynamic historical Daylight Saving Time (DST) offsets (-8h PST / -7h PDT).
   - Flagged 234 batch-upload end-date anomalies where stations reconnected to network servers hours after session termination.
3. **Port Type Recovery:** Imputed 9 missing `Port Type` values by establishing an unambiguous mapping from `Plug Type` (`J1772` $\rightarrow$ `Level 2`).
4. **Causal Occupancy & Duration Derivation:** Derived `occupancy_start_utc`, `occupancy_end_utc`, and `active_charging_end_utc` for every session, flagging 45 anomalous rows where active charging time exceeded total dwell time.
5. **Ground Truth Reconciliation:** Total energy across all 259,411 cleaned sessions was verified as exactly **2,216,493.71 kWh (2,216.49 MWh / 2.216 GWh)**.

---

## 9. Exploratory Data Analysis (Phase 2)

Key empirical findings from Phase 2 exploratory visualizations:
- **Diurnal Commuter Wave:** Network arrivals rise sharply at 06:00, peak at **11:00 PT (31,529 sessions, 12.15%)**, and decline through the afternoon.
- **Weekday Dominance:** Weekdays average **85.3 sessions/day network-wide** vs. **46.8 sessions/day on weekends (-45.1%)**.
- **Energy Timing Shift:** Peak energy delivery occurs at **09:00 PT (171.66 MWh)**, two hours prior to the arrival count peak, due to deeper battery depletion among early commuters.
- **Port Usage Disparity:** Level 2 chargers deliver **99.35% of all energy (2,202,139.80 kWh)** across 254,730 sessions. Level 1 chargers account for only 4,681 sessions and 14,353.92 kWh.
- **Station Heterogeneity:** Hamilton #2 is the highest-volume station network-wide (23,721 sessions; 191.72 MWh), whereas newly opened sites in 2020 recorded fewer than 10 sessions.

---

## 10. Feature Engineering & Contiguous Panel Construction (Phase 4)

To train supervised ML models, session logs were aggregated into a contiguous hourly station-level panel (`dataset/processed/ev_charging_hourly_features.csv`, 1,533,050 rows $\times$ 44 columns):
- **Contiguous Time Grid:** For each of the 46 stations, hourly intervals were constructed starting from the station's first observed session through 2020-12-31. Hours with no arrivals were explicitly recorded as `sessions_count = 0`.
- **Target Variable:** `sessions_count` (integer count of session arrivals in that station-hour).
- **Engineered Feature Categories:**
  1. *Calendar & Cyclical:* `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `dow_sin`, `dow_cos`, `month`, `month_sin`, `is_weekend`, `is_us_federal_holiday`, `is_working_day`.
  2. *Station Activity & Age:* `station_encoded`, `station_age_days`, `station_cum_sessions_prior`, `station_expanding_mean_sessions`, `station_hour_expanding_mean_sessions`.
  3. *Causal Lags (1h Horizon):* `lag_1h_sessions`, `lag_2h_sessions`, `lag_3h_sessions`, `lag_24h_sessions`, `lag_168h_sessions`.
  4. *Causal Lags (24h Day-Ahead Horizon):* Lags strictly $\le t - 24\text{h}$ (`lag_24h`, `lag_48h`, `lag_168h`, `lag_336h`).
  5. *Rolling Windows:* `rolling_mean_24h`, `rolling_sum_24h`, `rolling_mean_168h`, `rolling_sum_168h`.
  6. *Pre-Hour Concurrency:* `prior_plugged_in_vehicles_at_hour_start`, `prior_active_charging_vehicles_at_hour_start`.

---

## 11. Machine Learning Methodology & Anti-Leakage Design

```
CHRONOLOGICAL DATA SPLIT TIMELINE:
[2011 ────────────── 2017] [   2018   ] [      2019      ] [         2020         ]
    Training Split          Validation      Main Test Set       COVID-19 Stress Test
  (707,668 station-hrs)   (277,536 hrs)    (289,080 hrs)        (258,766 hrs)
                          Tune & Select     Held-Out Eval     Out-of-Distribution Eval
                               │
                Retrain on 2011–2018 (985,204 hrs)
```

### Strict Anti-Leakage Protocol
1. **No Temporal Shuffling:** Chronological ordering was strictly maintained.
2. **Causal Preprocessing:** Standard scalers, frequency encoders, and missing value indicators were fitted solely on training splits.
3. **Day-Ahead Information Cutoff:** For 24-hour predictions, all features for target hour $t_k = T_{\text{last}} + k$ are computed using data strictly $\le t_k - 24\text{h} \le T_{\text{last}}$.

---

## 12. Model Training & Evaluation (Phase 5)

Evaluated algorithms included:
- **Baseline Models:** Global Constant Mean, Historical Hour-of-Week Mean, Seasonal Naive (Lag 168h).
- **Linear Models:** Ridge Regression with L2 regularization.
- **Tree Ensembles:** Random Forest Regressor and XGBoost Regressor.

### Evaluation Metrics
- **Root Mean Squared Error (RMSE):** Primary metric penalizing large commuter-peak errors:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **Mean Absolute Error (MAE):** Average magnitude of arrival count errors:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Coefficient of Determination ($R^2$):** Proportion of variance explained relative to mean baseline.
- **Volume Calibration Bias (%):** Percentage deviation between total predicted and actual arrivals.

---

## 13. Results & Discussion

### 13.1 Main Test Set Performance (2019)
On the held-out 2019 test set (289,080 station-hours), **XGBoost achieved the best performance across all primary metrics**:
- **1-Hour-Ahead Model:**
  - **RMSE:** `0.3564` (vs. `0.3845` Historical Mean, `0.4439` Seasonal Naive, `0.3604` Random Forest).
  - **MAE:** `0.2029` (vs. `0.2455` Historical Mean, `0.2115` Random Forest).
  - **$R^2$:** `0.2723` (vs. `0.1531` Historical Mean, `0.2562` Random Forest).
  - **Volume Bias:** `-3.43%` (46,092 predicted vs. 47,731 actual sessions).
- **24-Hour-Ahead Model:**
  - **RMSE:** `0.3629` (vs. `0.3845` Historical Mean, `0.3671` Random Forest).
  - **MAE:** `0.2164` (vs. `0.2455` Historical Mean, `0.2223` Random Forest).
  - **$R^2$:** `0.2458` (vs. `0.1531` Historical Mean, `0.2285` Random Forest).

### 13.2 Out-of-Distribution COVID-19 Stress Test (2020)
In 2020, severe shelter-in-place orders altered driving habits:
- **Overall 1-Hour Results:** **Random Forest achieved a slightly lower overall RMSE (`0.2661` vs. `0.2696`)** because its shallow bagging structure regularized better against the network-wide demand collapse.
- **Cold-Start Sherman Stations:** On the 13 newly opened Sherman Avenue stations, **XGBoost decisively won over Random Forest**, achieving an RMSE of `0.0445` (vs. `0.0559`) and an $R^2$ of **`+0.2423`** (vs. `-0.1974` for Random Forest).
- **Scientific Conclusion:** XGBoost is superior in normal operating regimes and generalizes effectively to new station spatial topologies, while Random Forest provides strong defense against unmodeled macro-economic demand drops.

---

## 14. Web Application Architecture (Phase 6)

The web dashboard is architected as a lightweight, production-grade web system:
- **Backend:** Flask application structured with modular Blueprints (`/api/dashboard`, `/api/stations`, `/api/predictions`, `/api/recommendations`). Models are loaded into process memory once via a singleton service.
- **Frontend:** React 18 single-page application built with Vite 5, Tailwind CSS, and Recharts.
- **Deployment Modality:** Unified production mode where Flask hosts both the REST API and the compiled React static bundle from a single server process on port 5000.

---

## 15. Application Modules

1. **Network Overview (Dashboard):** Summarizes historical totals (259,411 sessions, 2,216.49 MWh), weekday vs. weekend 24h curves, Level 1 vs. Level 2 statistics, and top stations.
2. **Demand Forecasts (Predictions):** Interactive station-level 1h and 24h forecasting, expected Poisson arrival rates, rounded session counts, and historical actual-vs-predicted backtesting.
3. **Station Explorer:** 46-station searchable directory with operating dates, dwell times, and cold-start status badges.
4. **Optimization Plan (Recommendations):** Categorized rule-based planning guidance with priority tags and capacity disclaimers.
5. **Model Evaluation:** Full benchmark documentation across Validation 2018, Test 2019, and OOD 2020.

---

## 16. Optimization Recommendation Methodology

The recommendation engine translates statistical forecasts into non-prescriptive municipal planning guidance:
- **Category 1 – High Predicted Demand (1h):** Triggered when $p_{\text{1h}} \ge 0.25$ and $p_{\text{1h}} \ge 1.30 \times \text{hist\_avg}$.
- **Category 2 – Day-Ahead Planning (24h):** Top 5 established stations by 24h predicted volume.
- **Category 3 – Recurring Peak Demand:** Station peak hour mean $\ge 0.45$/hr with peak-to-average ratio $\ge 1.90$.
- **Category 4 – Low Historical Demand:** Established stations averaging $< 2.2$ sessions/day.
- **Category 5 – Cold-Start Monitoring:** Newly opened Sherman Avenue stations requiring initial data collection.
- **Capacity Disclaimer:** Every payload includes the required notice that recommendations reflect arrival flows and do not indicate confirmed physical charger congestion.

---

## 17. Testing & Verification

Comprehensive automated testing via `scripts/test_backend_api.py` confirmed:
- 11 / 11 automated test suites passed.
- $0.000000$ difference against standalone `ml/predict_demand.py` inference.
- Exact match of dashboard energy totals (`2,216,493.71 kWh`) against the cleaned CSV.
- Cryptographic SHA-256 verification confirmed all upstream datasets and model binaries remained 100% untouched.

---

## 18. Limitations & Methodological Constraints

1. **Target Variable Boundary:** The model predicts vehicle arrival count (`sessions_count`), not concurrent parking space occupancy or electrical power load.
2. **Missing Infrastructure Telemetry:** The City of Palo Alto dataset does not record physical port counts, queue lengths, or real-time maintenance outages. Recommendations cannot be construed as confirmed hardware shortages.
3. **Cold-Start Uncertainty:** Stations opened in December 2020 have minimal history; forecasts for these sites rely on generalized network features and should be monitored as telemetry accumulates.

---

## 19. Future Enhancements

1. **Real-Time API Integration:** Connect live ChargePoint or Open Charge Point Protocol (OCPP) APIs to ingest real-time port occupancy and pricing signals.
2. **Dynamic Pricing Modules:** Implement tariff optimization algorithms to incentivize off-peak charging and flatten commuter peaks.
3. **Solar PV & Battery Integration:** Model collocated solar generation and battery storage dispatch to minimize grid demand charges.

---

## 20. Conclusion

This project successfully designed, implemented, and verified an end-to-end Machine Learning and Decision Support System for municipal EV charging station demand optimization. By enforcing strict chronological data splits, constructing a leak-free 1.5-million-hour feature panel, and benchmarking XGBoost against linear and tree ensembles, the system delivers verified arrival forecasts and explainable planning recommendations through a high-performance web dashboard.

---

## 21. References

1. City of Palo Alto. (2021). *Electric Vehicle Charging Station Usage Dataset*. Open Data Palo Alto.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Al-Hanahi, B., Ahmad, I., Habibi, D., & Masoum, M. A. (2021). *Charging Infrastructure for Commercial Electric Vehicles: Challenges and Future Works*. IEEE Access, 9, 121476-121492.
5. Zhang, X., & Wang, Y. (2022). *Short-Term Electric Vehicle Charging Demand Forecasting Based on Deep Learning and Spatio-Temporal Features*. Applied Energy, 305, 117865.
