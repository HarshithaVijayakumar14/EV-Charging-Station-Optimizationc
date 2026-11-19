# Electric Vehicle Charging Station Optimization
## Predicting Demand for EV Stations (City of Palo Alto, CA)

An end-to-end Machine Learning and Decision Support system for municipal EV charging network optimization. Integrates cleaned historical charging telemetry (2011–2020) with pre-trained XGBoost models to provide leakage-free 1-hour-ahead and 24-hour day-ahead session arrival predictions, interactive network analytics, and explainable rule-based demand-planning recommendations.

---

## Key Features

- **Network-Wide Analytics:** Real-time exploration of 259,411 historical charging sessions, 2.216 GWh (2,216.49 MWh) delivered energy, 46 charging stations, Level 1 vs. Level 2 comparisons, and weekday vs. weekend commute profiles.
- **Dual-Horizon Forecasting:**
  - **1-Hour-Ahead Arrival Forecasts:** Real-time expected arrival rates ($p_f$ Poisson rate and rounded session count) for immediate operational awareness.
  - **24-Hour Day-Ahead Planning:** 24 individual hourly step predictions using strictly causal features ($\le t - 24\text{h}$) to forecast peak arrival hours and daily arrival totals.
- **Historical Backtest Mode:** Visualizes actual vs. predicted session arrivals over the last 48h, 72h, or 168h of observed history with window Mean Absolute Error (MAE) benchmarks.
- **Station Explorer:** Detailed directory and telemetry view for all 46 stations, including address, geographic coordinates, operating spans, dwell times, and clear indicators for newly commissioned cold-start sites.
- **Explainable Rule-Based Optimization Engine:** Generates transparent, categorized planning recommendations based on model forecasts and historical demand concentration, backed by a prominent capacity limitation disclaimer.
- **Model Performance Benchmarking:** Full visibility into Phase 5 validation (2018), test (2019), and COVID-19 out-of-distribution (2020) evaluation metrics, feature permutation importance, and calibration curves.
- **Intelligent EV Charging Assistant (Phase 8):**
  - **Physical Charging Point Inventory:** Verified hardware audit of all 120 physical points across 46 stations, distinguishing active equipment from decommissioned/upgraded pedestals.
  - **Hourly Availability Reconstruction:** Complete historical availability dataset satisfying $\text{occupied} + \text{free} + \text{unknown} \equiv \text{observed}$ across all 1,533,050 station-hours.
  - **Future Availability Forecaster:** Erlang-B loss model integrating XGBoost arrival forecasts with dwell times to estimate expected open ports and occupancy risk.
  - **Vehicle Catalog & Compatibility:** 15 verified OEM electric vehicles (four-wheelers, two-wheelers, custom) with J1772, NACS adapter, and NEMA 5-20R Level 1 rules.
  - **Battery CC/CV Taper Charging Estimator:** Accurate duration modeling reflecting constant-current bulk ($<80\%$) and constant-voltage thermal taper ($80\%\text{--}100\%$) phases.
  - **Nearby Station Recommendation Engine:** Multi-factor ranking by geodesic distance, compatibility, available ports, and charging speed within 5 miles.
- **Dual-Mode Deployment:** Unified single-command server hosting both backend API and pre-compiled React SPA on port 5000, or modular development setup with Vite hot-reloading on port 5173.

---

## Project Structure

```
├── backend/                        # Flask REST API backend
│   ├── app.py                      # App factory, static SPA routing, CORS, health checks
│   ├── config.py                   # Paths, constants, and server configuration
│   ├── database/
│   │   ├── db.py                   # SQLite schema, indexing, and prediction logging
│   │   └── ev_app.db               # SQLite database file
│   ├── routes/
│   │   ├── dashboard.py            # /api/dashboard/summary
│   │   ├── stations.py             # /api/stations, /api/stations/nearby, /api/stations/<id>
│   │   ├── predictions.py          # /api/predictions/1-hour, /24-hour, /station/<id>
│   │   ├── recommendations.py      # /api/recommendations
│   │   └── charging_assistant.py   # /api/charging-points/*, /api/availability/*, /api/vehicles, /api/charging/estimate, /api/stations/recommend
│   └── services/
│       ├── model_service.py        # Singleton loader for Phase 5 XGBoost models
│       ├── data_service.py         # Read-only cached analytics and tail panels
│       ├── prediction_service.py   # Leakage-free 1h & 24h forecasting engine
│       └── recommendation_service.py # Explainable rule-based planning engine
├── data/
│   └── vehicle_catalog.csv         # 15 verified electric vehicle models & OEM specs
├── dataset/
│   ├── raw/                        # Original raw CSV dataset (read-only)
│   └── processed/
│       ├── ev_charging_cleaned.csv # Cleaned session dataset (259,411 rows)
│       ├── ev_charging_hourly_features.csv # Contiguous hourly feature panel (1,533,050 rows)
│       ├── charging_point_inventory.csv    # Verified 120 physical charging points
│       └── historical_station_availability.csv # Hourly availability with mathematical conservation
├── frontend/                       # React 18 + Vite + Tailwind CSS Single-Page App
│   ├── dist/                       # Compiled production frontend bundle
│   ├── src/
│   │   ├── components/             # StatCard, Sidebar, Navbar, StationBadge, etc.
│   │   ├── pages/                  # Dashboard, FindStation, Predictions, StationAnalysis, Recommendations, ModelPerformance
│   │   ├── services/api.js         # Axios API client (full Phase 6 & Phase 8 endpoints)
│   │   ├── App.jsx                 # Application layout and navigation
│   │   └── main.jsx                # React root entry point
│   └── package.json                # Frontend dependencies
├── ml/                             # Preprocessing, model training & availability scripts
│   ├── common.py                   # Anti-leakage feature sets and holidays
│   ├── train_model.py              # Chronological training pipeline
│   ├── evaluate_model.py           # Evaluation on 2018, 2019, and 2020
│   ├── predict_demand.py           # Standalone batch inference script
│   ├── charging_point_inventory.py # Hardware inventory reconstruction
│   ├── historical_availability.py  # Historical availability ledger
│   ├── availability_prediction.py  # Future Erlang-B availability estimator
│   ├── charging_time_estimator.py  # CC/CV battery taper duration estimator
│   ├── station_compatibility.py    # Vehicle-station connector compatibility
│   └── station_recommender.py      # Geodesic nearby alternative recommender
├── models/                         # Pre-trained models and metadata
│   ├── best_model_1h.joblib        # Retrained 1-hour XGBoost model bundle
│   ├── best_model_24h.joblib       # Retrained 24-hour XGBoost model bundle
│   └── training_metadata.json      # Training parameters and validation benchmarks
├── reports/                        # Detailed project documentation and reports
│   ├── phase1_dataset_analysis_report.md
│   ├── phase2_eda_report.md
│   ├── phase3_cleaning_report.md
│   ├── phase4_feature_engineering_report.md
│   ├── phase5_model_evaluation_report.md
│   ├── phase6_application_report.md
│   ├── phase8_inspection_report.md
│   ├── phase8_availability_report.md
│   ├── eda_visualizations/         # 8 charts from Phase 2
│   └── model_evaluation/           # 6 charts from Phase 5
├── scripts/
│   ├── test_backend_api.py         # Phase 6 automated regression suite (all 11 tests pass)
│   └── test_availability.py        # Phase 8 automated test suite (all 14 tests pass)
└── requirements.txt                # Python dependencies
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10 to 3.12 (Python 3.12 recommended)
- **Node.js:** v18+ (tested on Node v24.19.0)
- **npm:** v9+

### 1. Python Environment Setup
```bash
# Clone the repository and navigate to root directory
cd ML

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run build   # Compiles production bundle to frontend/dist/
cd ..
```

---

## How to Run

### Option A: Unified Production Mode (Recommended)
Runs both the Flask API and the pre-built React frontend from a single server:
```bash
.\.venv\Scripts\activate
python backend/app.py
```
Open **`http://localhost:5000`** in your browser.

### Option B: Active Development Mode
Enables hot-module replacement for frontend edits:

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
Open **`http://localhost:5173`** in your browser.

---

## Running Automated Tests

Run the complete verification test suites to audit API health, numerical parity with `ml/predict_demand.py`, mathematical conservation, and intelligent availability forecasting:

**Phase 6 Backend Regression Suite:**
```bash
.\.venv\Scripts\activate
python scripts/test_backend_api.py
```

**Phase 8 Availability & Recommendation Test Suite:**
```bash
.\.venv\Scripts\activate
python scripts/test_availability.py
```

---

## API Reference Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check, model readiness, and SQLite count |
| `/api/dashboard/summary` | `GET` | Network totals, energy, dates, Level 1/2 stats, top 10 stations |
| `/api/history/hourly` | `GET` | 24h arrival curves (overall, weekday vs. weekend) |
| `/api/history/daily` | `GET` | Daily arrival counts, active stations, monthly aggregates |
| `/api/stations` | `GET` | List of all 46 stations with summary metadata and cold-start flags |
| `/api/stations/nearby` | `GET` | Stations within search radius sorted by geodesic distance (miles & km) |
| `/api/stations/<station_id>` | `GET` | Granular station telemetry, peak hours, dwell time statistics |
| `/api/predictions/1-hour` | `GET` | Next-hour arrival forecast ($T_{\text{last}} + 1\text{h}$) across all stations |
| `/api/predictions/24-hour` | `GET` | Next 24 hours of hourly predictions using features $\le t - 24\text{h}$ |
| `/api/predictions/station/<id>` | `GET` | Station forecast (1h and 24h) + historical Actual vs. Predicted backtest |
| `/api/predictions/history` | `GET` | Audit log of logged predictions from SQLite |
| `/api/recommendations` | `GET` | Rule-based demand planning suggestions and operational rationales |
| `/api/model/performance` | `GET` | Actual Phase 5 evaluation benchmarks, feature importance, calibration |
| `/api/charging-points/inventory` | `GET` | Verified 120 physical charging points across all 46 stations |
| `/api/charging-points/station/<id>` | `GET` | Physical hardware inventory, active ports, and lifetime turnover for station |
| `/api/availability/historical` | `GET` | Historical hourly availability records with mathematical conservation |
| `/api/availability/predict` | `GET` | Future availability forecast (expected occupied, estimated free, risk of full station) |
| `/api/vehicles` | `GET` | Catalog of 15 verified electric vehicle models & technical specifications |
| `/api/charging/estimate` | `POST` | Calculates charging duration with battery CC/CV taper curves above 80% SOC |
| `/api/stations/recommend` | `POST` | Assesses target station and recommends nearby alternatives ranked by composite score |

---

## Methodological & Planning Disclaimers

1. **Prediction Target:** The target variable is `sessions_count`, representing the number of new vehicle charging arrivals during that clock hour. It does not measure concurrent charger occupancy or remaining vacant parking spaces.
2. **Capacity Information:** The City of Palo Alto dataset does not include physical charger port counts, queue lengths, vehicle turn-away rates, or maintenance outage logs. Predictions and recommendations are non-prescriptive planning tools and must not be interpreted as confirmed congestion or physical capacity deficits.
3. **Cold-Start Stations:** The 13 Sherman Avenue stations were commissioned in December 2020 during the COVID-19 pandemic. Five of these stations have fewer than 24 hours of history and are safely withheld from lag-based forecasts until sufficient operating telemetry is collected.

---

## License & Credits
Developed as part of the *Electric Vehicle Charging Station Optimization* project using public EV charging data provided by the City of Palo Alto, California.
