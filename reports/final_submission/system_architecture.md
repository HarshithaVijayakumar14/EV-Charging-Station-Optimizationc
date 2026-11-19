# System Architecture & Technical Specifications
## Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations

**Project:** City of Palo Alto EV Charging Demand Prediction & Planning Decision Support System  
**Document:** System Architecture & Data Flow Specification  
**Version:** 1.0 (Final Submission)  
**Date:** November 2026  

---

## 1. Architectural Overview

The project is structured as an end-to-end, multi-stage machine learning and software engineering pipeline designed for municipal EV charging network optimization. The architecture bridges raw session-level transaction telemetry with high-performance regression models and delivers actionable arrival-rate predictions and planning recommendations through a modern web application.

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA PROCESSING PIPELINE                                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Raw Data: 259,415 records x 33 cols]                                                             |
|                         │                                                                          |
|                         ▼ Phase 3: Data Cleaning & Preprocessing                                   |
|  [Cleaned Dataset: 259,411 records x 78 cols; 2,216,493.71 kWh]                                   |
|  - Deduplication (-4 full-row duplicates)                                                          |
|  - Timezone standardisation (Pacific Time with accurate DST offsets)                               |
|  - Port type recovery (9 unambiguous mappings from Plug Type)                                      |
|  - Causal occupancy & charging interval derivation                                                |
|                         │                                                                          |
|                         ▼ Phase 4: Contiguous Hourly Feature Panel Construction                    |
|  [Hourly Features: 1,533,050 station-hours x 44 cols]                                              |
|  - Full temporal grid spanning 46 stations from individual commissioning to 2020-12-31             |
|  - Strictly causal lags (1h, 2h, 3h, 24h, 168h) & rolling statistics (24h, 168h)                   |
|  - Expanding station historical means & calendar harmonic features                                 |
|  - Pre-hour concurrent vehicle counts at station                                                   |
+----------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                   MODEL TRAINING & EVALUATION                                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Chronological Data Splits:                                                                        |
|  ├── Train: 2011–2017 (707,668 station-hours; 27 historical stations)                              |
|  ├── Validation: 2018 (277,536 station-hours; 27 historical + 6 new stations)                      |
|  ├── Retrain Set: 2011–2018 (985,204 station-hours; 33 stations)                                   |
|  ├── Main Test: 2019 (289,080 station-hours; 33 historical stations)                              |
|  └── OOD Stress Test: 2020 (258,766 station-hours; 33 historical + 13 new Sherman sites)           |
|                                                                                                    |
|  Trained Artifacts (Joblib):                                                                       |
|  ├── `models/best_model_1h.joblib`  (XGBoost 1-Hour-Ahead Regressor; 39 features)                 |
|  ├── `models/best_model_24h.joblib` (XGBoost 24-Hour-Ahead Regressor; 23 features, <= t-24h causal) |
|  └── `models/training_metadata.json` & `evaluation_results.json`                                   |
+----------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                     BACKEND SERVICES (FLASK)                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Flask Application Factory: `backend/app.py`]                                                     |
|  - CORS Enabled (`flask-cors`)                                                                     |
|  - Static Single-Page App (SPA) Routing & Fallback                                                 |
|  - Centralized Error Handling & Health Monitoring (`/api/health`)                                  |
|                                                                                                    |
|  Modular Blueprints:                                                                               |
|  ├── `backend/routes/dashboard.py`       --> `/api/dashboard/summary`                             |
|  ├── `backend/routes/stations.py`        --> `/api/stations`, `/api/stations/<id>`                 |
|  ├── `backend/routes/predictions.py`     --> `/api/predictions/1-hour`, `/24-hour`, `/station/<id>`|
|  ├── `backend/routes/recommendations.py` --> `/api/recommendations`                                |
|  ├── `backend/routes/probabilistic.py`   --> `/api/probabilistic/arrivals`, `/availability`, ...   |
|  └── `backend/routes/charging_assistant.py` --> `/api/charging-points/*`, `/api/availability/*`   |
|                                                                                                    |
|  Core Domain Services:                                                                             |
|  ├── `ModelService`: Singleton model & preprocessor bundle loader (loaded once at app start)       |
|  ├── `ProbabilisticService`: Singleton loader for count PMFs, scoring rules & Erlang propagation   |
|  ├── `DataService`: Fingerprinted derived cache (`dashboard_data_cache.joblib`), station tails    |
|  ├── `PredictionService`: Zero-leakage causal feature assembler, 1h/24h scoring, backtest engine   |
|  └── `RecommendationService`: Rule-based explainable planning suggestions with capacity disclaimer|
|                                                                                                    |
|  Persistence Layer:                                                                                |
|  └── SQLite Database (`backend/database/ev_app.db`): `prediction_history`, `recommendation_history` |
+----------------------------------------------------------------------------------------------------+
                                               │
                                 JSON REST API │ HTTP (Port 5000 / 5173)
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                     FRONTEND CLIENT (REACT)                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Single-Page Application: React 18 + Vite 5 + Tailwind CSS]                                       |
|                                                                                                    |
|  Modular Navigation & Views:                                                                       |
|  ├── Dashboard (`DashboardPage.jsx`): Network KPIs, 24h weekday/weekend curves, Level 1 vs Level 2 |
|  ├── Predictions (`PredictionsPage.jsx`): Station selector, 1h & 24h forecasts, historical backtest|
|  ├── Station Explorer (`StationAnalysisPage.jsx`): 46-station directory, dwell vs charge durations |
|  ├── Recommendations (`RecommendationsPage.jsx`): 5-category planning cards, capacity disclaimers |
|  ├── Model Evaluation (`ModelPerformancePage.jsx`): Val/Test/OOD benchmarks, permutation importance|
|  └── Probabilistic Availability (`ProbabilisticAvailabilityPage.jsx`): Discrete PMFs & Erlang-B UQ |
|                                                                                                    |
|  Chart Engine: Recharts (Responsive SVG Area, Bar, Line, and Composed charts)                      |
|  Design System: Tailwind CSS corporate palette, Lucide React icons, mobile-responsive sidebar      |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Data Processing & Storage Layer
- **Input Data:** `dataset/raw/EVChargingStationUsage.csv` (85.45 MB raw CSV, 259,415 records, 33 columns). Preserved read-only with cryptographic SHA-256 verification.
- **Processed Cleaned CSV:** `dataset/processed/ev_charging_cleaned.csv` (212.75 MB, 259,411 rows, 78 columns). Cleaned dates, standardized timezones, recovered port types, and duration flags.
- **Contiguous Hourly Panel:** `dataset/processed/ev_charging_hourly_features.csv` (448.09 MB, 1,533,050 rows, 44 columns). Represents contiguous hourly intervals across all 46 stations from each station's commissioning date through 2020-12-31.
- **Derived Cache:** `backend/cache/dashboard_data_cache.joblib` (11.7 MB). Built once upon startup from Phase 3 & 4 CSVs. Employs filesystem metadata fingerprinting (`mtime` + `size` check). Prevents repeated multi-hundred-megabyte disk reads on API requests, ensuring sub-20ms dashboard response times.

### 2.2 Machine Learning Engine Layer
- **Preprocessors:** Custom `StationFeaturePreprocessor` objects bundled inside joblib files. Implements target/frequency station encoding, cyclical feature transformations, standard scaling, and missing value indicators fitted exclusively on chronological training splits.
- **Model Bundle 1-Hour-Ahead (`models/best_model_1h.joblib`):**
  - **Algorithm:** XGBoost Regressor (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
  - **Feature Count:** 39 causal features including calendar harmonics, expanding station-hour means, concurrent plugged-in vehicles at hour start, and 1h–168h lags.
  - **Execution Constraint:** Zero future leakage; evaluates strictly at forecast origin $T_{\text{last}}$.
- **Model Bundle 24-Hour-Ahead (`models/best_model_24h.joblib`):**
  - **Algorithm:** XGBoost Regressor (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
  - **Feature Count:** 23 day-ahead features.
  - **Execution Constraint:** Strictly causal day-ahead cutoff ($\le t - 24\text{h}$). For each step $k \in [1, 24]$, all features are drawn from history $\le t_k - 24\text{h} \le T_{\text{last}}$. No recursive synthetic predictions are fed into subsequent steps.

### 2.3 Backend API Layer (Python Flask)
- **Framework:** Flask 3.1.0 with Flask-CORS 5.0.0.
- **Execution Mode:** Single-threaded or multi-threaded WSGI.
- **Model Lifecycle:** Singleton pattern via `backend/services/model_service.py`. Models are deserialized once into process memory during `create_app()` and reused across requests.
- **Database (`backend/database/db.py`):** Lightweight SQLite3 engine (`backend/database/ev_app.db`). Tables:
  - `prediction_history`: Logs forecast origin, target timestamp, predicted session count, station ID, and horizon.
  - `recommendation_history`: Logs generated planning recommendations, category, priority, and quantitative metrics.

### 2.4 Frontend Layer (React / Vite)
- **Framework:** React 18.3.1 with JavaScript (ES Modules).
- **Bundler:** Vite 5.4.11 with `@vitejs/plugin-react`.
- **Styling:** Tailwind CSS 3.4.15 with Autoprefixer and PostCSS.
- **Charting Library:** Recharts 2.13.3 (SVG-based reactive visualization engine).
- **Icons:** Lucide React 0.460.0.
- **Client Routing:** Responsive state-based tab router with URL synchronization and SPA fallback support on Flask.

---

## 3. Data Flow & Communication Protocols

### 3.1 Prediction Request Lifecycle
1. User selects a charging station (e.g. `PALO ALTO CA / HAMILTON #2`) and horizon (1h or 24h) on the web dashboard.
2. The React client issues an asynchronous HTTP GET request: `GET /api/predictions/station/PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232?backtest_hours=72`.
3. Flask routes the request to `backend/routes/predictions.py`, which delegates to `PredictionService`:
   - `PredictionService` fetches station metadata and the station's recent 504-hour causal panel from `DataService`.
   - Causal features are dynamically extracted strictly $\le T_{\text{last}}$ (for 1h) and strictly $\le t - 24\text{h}$ (for 24h).
   - `ModelService` applies the bundled preprocessor and executes `model.predict(X)`.
   - Predictions are non-negatively clipped (`np.clip(p, 0.0, None)`), rounded, and contextualized against the station's historical same-hour average.
   - For backtesting, the observed past window is scored against known actuals and window MAE is computed.
   - Prediction results are logged to SQLite (`prediction_history`).
4. Flask serializes the payload to JSON and returns HTTP 200 OK.
5. The React client receives the payload and updates the interactive ComposedChart (bar + line overlay) and metric stat cards in real time.

---

## 4. Security, Isolation & Anti-Leakage Safeguards

1. **Read-Only Data Isolation:** Upstream datasets (`EVChargingStationUsage.csv`, `ev_charging_cleaned.csv`, `ev_charging_hourly_features.csv`) are accessed strictly in read-only mode (`r` flag in Python, low-memory chunking). Checksums are programmatically verified before and after test suites.
2. **Chronological Time Boundaries:** No observation from validation year 2018, test year 2019, or OOD year 2020 was ever made accessible during model training. In real-time inference, feature generation strictly enforces causality ($t_{\text{feature}} \le T_{\text{forecast\_origin}}$).
3. **No Cross-Station Memory Bleed:** Feature assembly resets station identifiers and slices only the designated station's contiguous tail panel.
4. **Cold-Start Safeguards:** Stations with fewer than 24 recorded hours cleanly exit with `insufficient_history` rather than imputing artificial zero lags or producing hallucinated forecasts.
5. **Planning Boundary Disclaimers:** All recommendation cards, API recommendation payloads, and dashboard prediction curves explicitly carry the methodological disclaimer that the target represents session arrival flow and does not measure simultaneous charger occupancy or physical capacity.
