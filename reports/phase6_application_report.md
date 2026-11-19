# Phase 6: Web Application Development Report
## Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations

**Project:** City of Palo Alto EV Charging Demand Prediction & Optimization Dashboard  
**Phase:** 6 — Web Application Development  
**Date:** November 2026  
**Status:** Complete & Fully Verified  

---

## 1. Executive Summary

Phase 6 successfully delivers a complete, production-grade web application integrating the machine learning models trained in Phase 5 with historical charging telemetry from Phases 1–4. The web dashboard provides municipal planners, fleet operators, and EV network managers with intuitive visibility into historical utilization patterns, real-time 1-hour-ahead arrival forecasts, 24-hour day-ahead demand projections, station-level telemetry, explainable rule-based demand-planning suggestions, and comprehensive model validation benchmarks.

### Core Objectives Achieved
1. **Interactive Analytics Dashboard:** Real-time presentation of network-wide statistics (259,411 sessions, 2.216 GWh / 2,216.49 MWh total energy, 46 charging stations, Level 1 vs. Level 2 breakdown, weekday vs. weekend temporal profiles).
2. **Pre-Trained ML Model Integration:** Embedded Phase 5 XGBoost regressors (`models/best_model_1h.joblib` and `models/best_model_24h.joblib`) loaded as singleton instances with zero in-app retraining.
3. **Dual-Horizon Forecasting:**
   - **1-Hour-Ahead Forecasts:** High-frequency arrival rate forecasting for immediate operational awareness.
   - **24-Hour-Ahead Forecasts:** Day-ahead planning horizon predicting expected arrival volumes across all 24 upcoming clock hours using strictly causal features ($\le t - 24\text{h}$).
4. **Historical Backtest Mode:** Allows users to inspect actual vs. predicted session arrivals over the last 48h, 72h, or 168h of observed history, displaying window Mean Absolute Error (MAE) and cumulative volume comparisons without data leakage.
5. **Station Explorer:** Searchable directory and telemetry view for all 46 stations, detailing geographic coordinates, address, active operating spans, dwell times, and clear indicators for newly commissioned cold-start sites.
6. **Explainable Rule-Based Optimization Engine:** Generates transparent, non-prescriptive planning recommendations categorized into 5 distinct operational themes, accompanied by a prominent capacity limitation disclaimer.
7. **Transparent Model Performance Documentation:** Surfaces actual Phase 5 evaluation benchmarks across Validation 2018, Main Test 2019, and the COVID-19 out-of-distribution (OOD) 2020 stress test, alongside permutation feature importance and calibration charts.
8. **Dual-Mode Deployment:** Supports both an active hot-reloading development workflow (Vite + Flask) and a single-command unified production server where Flask serves the pre-compiled React SPA directly.

---

## 2. System Architecture

```
                           +------------------------------------------+
                           |           User Browser / Client          |
                           |    (Desktop, Tablet, Mobile Responsive)  |
                           +--------------------+---------------------+
                                                |
                               HTTP / REST API  |  JSON / Static Assets
                                                v
                           +------------------------------------------+
                           |             Flask Web Server             |
                           |               (Port 5000)                |
                           |  - CORS & Error Handling Middleware      |
                           |  - Static Single-Page App (SPA) Hosting  |
                           +----+------------------+------------------+
                                |                  |
          +---------------------+                  +---------------------+
          |                                                              |
          v                                                              v
+-------------------------+                                    +-------------------------+
|   Blueprint API Routes  |                                    |   SQLite Database       |
| /api/dashboard/*        |                                    | (backend/database/      |
| /api/stations/*         |                                    |  ev_app.db)             |
| /api/predictions/*      |                                    | - prediction_history    |
| /api/recommendations    |                                    | - recommendation_history|
| /api/model/performance  |                                    +-------------------------+
+----+---------------+----+
     |               |
     v               v
+--------------------+----+                                    +-------------------------+
|     Services Layer      |                                    |  Pre-Trained ML Models  |
| - DataService           |<-----------------------------------|  (Phase 5 Joblib Files) |
| - ModelService          |                                    | - best_model_1h.joblib  |
| - PredictionService     |                                    | - best_model_24h.joblib |
| - RecommendationService |                                    | - training_metadata.json|
+----+---------------+----+                                    | - evaluation_results.json|
     |               |                                         +-------------------------+
     v               v
+--------------------+----+
| Read-Only Processed Data|
| - ev_charging_cleaned   |
| - ev_hourly_features    |
+-------------------------+
```

### Component Roles
- **Frontend (`frontend/`):** Built with React 18, Vite 5, Tailwind CSS, Lucide React icons, and Recharts. Engineered for sub-second responsiveness, clear typography, intuitive color-coded badges, and responsive layouts across all device sizes.
- **Backend API (`backend/`):** Flask application structured around modular Blueprints. Features read-only data streaming, singleton model loading, and zero in-memory prediction bleed across stations or time steps.
- **Data Caching Layer (`backend/services/data_service.py`):** Generates a derived read-only cache (`dashboard_data_cache.joblib`) from Phase 3 & 4 CSVs with cryptographic file mtime/size fingerprinting, allowing instantaneous API responses without repeatedly parsing 427 MB of raw feature files.
- **Audit Persistence (`backend/database/db.py`):** SQLite database (`ev_app.db`) logging forecast requests and generated planning recommendations with indexed timestamps for reproducibility and historical auditability.

---

## 3. Backend API Specification

All endpoints return standardized JSON structures with HTTP status codes `200 OK`, `400 Bad Request`, `404 Not Found`, or `500 Internal Server Error`.

### 3.1 System & Dashboard Endpoints
- `GET /api/health`
  - **Purpose:** Verifies operational health, model readiness, and database connection.
  - **Sample Response:**
    ```json
    {
      "status": "healthy",
      "service": "EV Charging Station Demand Prediction API",
      "models": {
        "models_ready": true,
        "model_1h_loaded": true,
        "model_24h_loaded": true,
        "model_1h_name": "XGBoost Regressor (1-Hour-Ahead)",
        "model_24h_name": "XGBoost Regressor (24-Hour-Ahead)"
      },
      "database": {
        "prediction_history_count": 83,
        "recommendation_history_count": 14
      }
    }
    ```
- `GET /api/dashboard/summary`
  - **Purpose:** Retrieves network totals, date ranges, Level 1 vs Level 2 breakdowns, and top 10 stations.
  - **Metrics Returned:** Total sessions (`259,411`), total energy (`2,216.49 MWh` / `2.216 GWh`), active stations (`46`), average session arrival rate (`0.1692` arrivals/station-hour).
- `GET /api/history/hourly`
  - **Purpose:** 24-hour network-wide arrival profile and weekday vs. weekend averages.
- `GET /api/history/daily?limit=365&year=2019`
  - **Purpose:** Time-series of daily sessions, active stations, and monthly aggregations.

### 3.2 Stations Endpoints
- `GET /api/stations`
  - **Purpose:** Returns listing of all 46 stations with summary metadata (total sessions, active days, sessions per active day, latitude/longitude, and cold-start classification).
- `GET /api/stations/<station_id>`
  - **Purpose:** Returns granular station telemetry, top peak hours, dwell time statistics, and recent daily activity records.

### 3.3 Prediction Endpoints
- `GET /api/predictions/1-hour?station_id=<id>&persist=true`
  - **Purpose:** Computes 1-hour-ahead session arrival predictions for all stations or a specified station for target hour $T_{\text{last}} + 1\text{h}$.
  - **Output Fields:** `predicted_sessions` (floating-point Poisson rate), `predicted_sessions_rounded` (nearest integer), `historical_hour_avg`, `prior_plugged_in_vehicles`.
- `GET /api/predictions/24-hour?station_id=<id>&persist=true`
  - **Purpose:** Computes 24 individual hourly step predictions for target hours $T_{\text{last}} + 1\text{h} \dots T_{\text{last}} + 24\text{h}$ using strictly causal features ($\le t - 24\text{h}$).
  - **Output Fields:** `total_predicted_sessions_24h`, `mean_hourly_predicted_sessions`, `peak_predicted_hour`, `hourly_predictions` array.
- `GET /api/predictions/station/<station_id>?backtest_hours=72`
  - **Purpose:** Combined payload providing both 1h and 24h predictions plus historical Actual-vs-Predicted backtest evaluations over the requested window.
- `GET /api/predictions/history?limit=50&horizon=1-hour`
  - **Purpose:** Audit log of persisted forecast predictions from SQLite.

### 3.4 Recommendations & Model Performance
- `GET /api/recommendations?category=<cat>&station_id=<id>`
  - **Purpose:** Evaluates rule-based planning logic across all stations and returns categorized planning suggestions with quantitative metric rationales.
- `GET /api/model/performance`
  - **Purpose:** Full Phase 5 evaluation artifact metadata (RMSE, MAE, R², bias %, sub-segment splits, feature importance rankings, and hourly calibration curves).

---

## 4. Forecasting & Anti-Leakage Methodology

### 4.1 Feature Construction & Exact Parity
The prediction pipeline (`backend/services/prediction_service.py`) strictly mirrors the feature transformations established in Phase 4 and Phase 5 (`ml/common.py` and `ml/predict_demand.py`):
- **1-Hour-Ahead Horizon:** Constructs calendar harmonics (`hour_sin`, `hour_cos`, `dow_sin`, `month_sin`), recent causal lags (`lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`, `lag_168h`), rolling statistics (`rolling_mean_24h`, `rolling_mean_168h`), and pre-hour concurrent vehicle counts (`prior_plugged_in_vehicles_at_hour_start`).
- **24-Hour-Ahead Horizon:** For each target step $k \in [1, 24]$, features are constructed using only data available at $t_k - 24\text{h} \le T_{\text{last}}$. Lags are strictly $24\text{h}, 48\text{h}, 168\text{h}, 336\text{h}$ prior to the target, completely eliminating recursive error compounding or future target leakage.

### 4.2 Handling Cold-Start & Insufficient-History Stations
Of the 46 stations in the Palo Alto dataset:
- **33 Stations:** Historical established stations (present in 2011–2018 training data).
- **13 Stations:** Newly installed Sherman Avenue stations commissioned in December 2020.
  - **8 Stations:** Have between 24 and 100 hours of history; the service generates forecasts while appending prominent `is_cold_start: true` flags and UI cautionary notices.
  - **5 Stations:** Have fewer than 24 hours of total history (`span_hours < 24`). Rather than producing invalid zero predictions or hallucinating missing features, the prediction engine safely returns `status: "insufficient_history"`, explaining that at least 24 hours of operating telemetry are required for causal lag calculations.

---

## 5. Explainable Rule-Based Planning Recommendations

The recommendation engine (`backend/services/recommendation_service.py`) derives transparent planning insights based on actual model outputs and verified historical distributions:

| Recommendation Category | Trigger Condition | Planning Intent | Priority |
| :--- | :--- | :--- | :--- |
| **High Predicted Demand (1h)** | $p_{\text{1h}} \ge 0.25$ and $p_{\text{1h}} \ge 1.30 \times \text{hist\_avg}$ | Real-time monitoring of unexpected arrival surges above typical baseline | High |
| **Day-Ahead Planning (24h)** | Top 5 established stations by total 24h forecasted arrivals | Next-day staffing, signage, or dynamic pricing prioritization | High / Med |
| **Recurring Peak Demand** | Station peak hour $\ge 0.45$/hr and peak/overall ratio $\ge 1.90$ | Long-term capital planning for recurring commute congestion | Medium |
| **Low Historical Demand** | Established stations averaging $< 2.2$ sessions per active day | Infrastructure review before deploying additional hardware | Low |
| **Cold-Start Data Monitoring** | Newly commissioned stations absent from training set | Data collection and model retraining recommendations | Info |

### Mandatory Capacity Disclaimer
Every recommendation payload and UI view prominently includes the required planning notice:
> **Planning Suggestion Only:** The City of Palo Alto dataset does not include verified station port capacities, real-time operational port counts, queue lengths, turn-aways, or maintenance logs. Recommendations reflect historical and predicted session arrival patterns and must not be interpreted as confirmed congestion, utilization percentages, or definitive infrastructure requirements.

---

## 6. Frontend User Experience & Pages

The application is organized into 5 cohesive views accessible via the persistent navigation sidebar:

1. **Network Overview (`DashboardPage.jsx`):**
   - KPI cards for total sessions, total energy, active sites, and peak/low hours.
   - 24-hour weekday vs. weekend arrival distribution curve.
   - Day-of-week bar chart highlighting weekday commuter concentration.
   - Multi-year monthly trend showing sustained growth through 2019 and the COVID-19 drop.
   - Level 1 vs. Level 2 energy and duration comparison.
   - Interactive ranking of top 10 stations with direct drill-down links.
   - Quick snapshot of next-hour top forecasted stations.
2. **Demand Forecasts (`PredictionsPage.jsx`):**
   - Station selector supporting all 46 sites with status badges (Established, Cold-Start, Insufficient History).
   - Horizon toggle between 1-Hour-Ahead and 24-Hour Day-Ahead forecasts.
   - Detailed expectation cards displaying float Poisson rates ($p_f$) alongside rounded session counts ($\approx n$).
   - Interactive 24-hour bar and line forecast curve comparing day-ahead predictions against historical same-hour averages.
   - Historical Backtest component plotting actual arrivals vs. 1h and 24h predictions over the last 48h–168h with window MAE metrics.
   - Network-wide comparison table allowing one-click switching between stations.
3. **Station Explorer (`StationAnalysisPage.jsx`):**
   - Searchable, filterable directory of all 46 stations.
   - Granular station cards showing street address, geographic coordinates, dwell time vs. active charging time, and operating dates.
   - Station-specific 24-hour arrival curve and recent daily session volume trends.
4. **Optimization & Planning (`RecommendationsPage.jsx`):**
   - Filterable feed of rule-based planning cards by category and priority.
   - Clear operational messages, quantitative ratios, metric tags, and direct links to run predictions for that station.
   - Prominent methodological disclaimer banner.
5. **Model Evaluation (`ModelPerformancePage.jsx`):**
   - Comprehensive documentation of Phase 5 training and evaluation metrics.
   - Benchmark tables comparing XGBoost against Random Forest, Ridge Regression, and Baseline methods across Validation 2018, Test 2019, and OOD 2020.
   - Top 12 feature permutation importance chart.
   - Test 2019 hourly calibration curve demonstrating near-perfect alignment with actual observed arrival distributions.
   - Educational glossary explaining RMSE, MAE, R² in sparse Poisson settings, and decimal arrival rates.

---

## 7. Verification & Quality Assurance

### 7.1 Automated Backend Test Suite
An automated end-to-end regression test suite was executed via `scripts/test_backend_api.py` covering 11 critical validation criteria:
```
Recording upstream SHA-256 checksums before backend test suite...
  [PASS] /api/health & both XGBoost models loaded once
  [PASS] /api/dashboard/summary matches actual cleaned dataset totals
  [PASS] /api/stations & /api/stations/<station_id>
  [PASS] Invalid station IDs return clean 404 errors
  [PASS] /api/predictions/1-hour & /api/predictions/24-hour (41 forecasted, 5 insufficient history)
  [PASS] Station-specific predictions, cold-start warnings, insufficient-history safety, no cache bleed
  [PASS] Exact numerical parity verified against ml/predict_demand.py (1h & 24h)
  [PASS] /api/history/*, /api/recommendations, /api/model/performance, & SQLite persistence
  [PASS] All upstream datasets & trained models verified 100% unmodified!

ALL BACKEND TESTS PASSED!
```

### 7.2 Numerical Parity Check
Predictions produced by the backend `PredictionService` were compared against standalone inference runs from `ml/predict_demand.py`:
- **1-Hour-Ahead Parity:** Absolute difference $= 0.000000$ (exact float match).
- **24-Hour-Ahead Parity:** Absolute difference $= 0.000000$ across all 24 individual forecast steps.

### 7.3 Frontend Compilation & Delivery
The React application was compiled via `npm run build --prefix frontend`:
- Built in `22.68s` with `0` syntax or bundling errors.
- Generated production bundle in `frontend/dist/`.
- Flask static routing test confirmed status `200 OK` on `/` with full HTML/JS/CSS delivery.

### 7.4 Cryptographic Data Integrity
All upstream data files and pre-trained models were verified to be strictly read-only and unmodified throughout Phase 6:
- `EVChargingStationUsage.csv`: Identical SHA-256 hash.
- `dataset/raw/EVChargingStationUsage.csv`: Identical SHA-256 hash.
- `dataset/processed/ev_charging_cleaned.csv`: Identical SHA-256 hash.
- `dataset/processed/ev_charging_hourly_features.csv`: Identical SHA-256 hash.
- `models/best_model_1h.joblib`: Identical SHA-256 hash.
- `models/best_model_24h.joblib`: Identical SHA-256 hash.

---

## 8. Deployment & Startup Guide

### 8.1 Unified Production Mode (Recommended)
In production mode, the Flask server hosts both the REST API and the compiled React single-page application from a single process:
```bash
# 1. Activate the Python virtual environment
.\.venv\Scripts\activate

# 2. Start the unified web application
python backend/app.py
```
Open **`http://localhost:5000`** in any modern web browser.

### 8.2 Hot-Reloading Development Mode
For active frontend development with instant Vite hot-module replacement:

**Terminal 1 (Backend API):**
```bash
.\.venv\Scripts\activate
python backend/app.py
```

**Terminal 2 (Frontend Dev Server):**
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser. Requests to `/api/*` are automatically proxied to the Flask server on port 5000.

---

## 9. Limitations & Practical Guidance

1. **Target Interpretation:** Predictions measure `sessions_count` (the number of new vehicles initiating a charging session during that hour). This is distinct from simultaneous vehicle occupancy or charger port availability.
2. **Missing Capacity Data:** Because physical port counts, queue lengths, and maintenance records are absent from the City of Palo Alto dataset, high arrival rates must not be conflated with confirmed port congestion.
3. **Cold-Start Site Caution:** The 13 Sherman Avenue stations opened during the COVID-19 period in December 2020. Forecasts for these sites carry higher uncertainty and should be re-evaluated as additional operating history accumulates.
