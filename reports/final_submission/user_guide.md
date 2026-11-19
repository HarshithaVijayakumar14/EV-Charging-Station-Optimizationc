# Installation & User Operating Guide
## Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations

**Project:** City of Palo Alto EV Charging Demand Prediction & Planning Decision Support System  
**Document:** End-to-End Installation, Deployment & User Manual  
**Version:** 1.0 (Final Submission)  
**Date:** November 2026  

---

## 1. System Requirements

### Hardware Requirements
- **CPU:** Dual-core 2.0 GHz or higher (Quad-core Intel Core i5 / AMD Ryzen 5 or better recommended).
- **RAM:** Minimum 8 GB RAM (16 GB recommended to smoothly handle multi-hundred-megabyte feature matrices during batch queries).
- **Disk Storage:** At least 2.5 GB free disk space (includes raw CSV, cleaned CSV, feature panel, joblib cache, and node_modules).
- **Display:** Responsive design supports standard desktops (1920×1080, 1366×768), laptops, tablets, and mobile browsers.

### Software Prerequisites
- **Operating System:** Windows 10/11, macOS (12.0+), or Linux (Ubuntu 20.04/22.04 LTS).
- **Python:** Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.0 64-bit).
- **Node.js & npm:** Node.js v18.0.0 or higher (Tested on Node v24.19.0 with npm 11.17.0).
- **Web Browser:** Modern browser with WebGL and SVG support (Google Chrome 110+, Mozilla Firefox 110+, Microsoft Edge 110+, or Safari 16+).

---

## 2. Installation & Setup

### Step 1: Open Terminal & Navigate to Project Root
Open PowerShell (on Windows) or Terminal (on Linux/macOS):
```powershell
# Navigate to the project root directory
cd c:\Users\harsh\OneDrive\Desktop\ML
```

### Step 2: Set Up Python Virtual Environment
If `.venv` is already created, activate it. Otherwise, create it first:
```powershell
# Create virtual environment (if not already present)
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# OR on Windows (Command Prompt):
# .\.venv\Scripts\activate.bat
# OR on Linux/macOS:
# source .venv/bin/activate
```

### Step 3: Install Python Dependencies
```powershell
pip install -r requirements.txt
```
*Key packages installed:* Flask, flask-cors, scikit-learn, xgboost, pandas, numpy, joblib.

### Step 4: Install Frontend Node Dependencies & Compile
```powershell
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Compile the production bundle
npm run build

# Return to project root
cd ..
```
*Note:* `npm run build` compiles the React application into `frontend/dist/` in approximately 20 seconds.

---

## 3. Starting the Application

The application supports two distinct execution modes:

### Mode A: Unified Production Server (Recommended for Evaluation & Demos)
In this mode, a single Flask server hosts both the complete REST API and the compiled React single-page application directly on port 5000:
```powershell
# Ensure virtual environment is active
.\.venv\Scripts\Activate.ps1

# Launch the unified server
python backend/app.py
```
**Access URL:** Open your browser and navigate to:  
👉 **`http://localhost:5000`**

### Mode B: Active Development Mode (Hot-Reloading)
For developers modifying the React UI or Tailwind CSS with instant hot-module replacement:

**Terminal 1 (Backend API):**
```powershell
.\.venv\Scripts\Activate.ps1
python backend/app.py
```
*(Runs Flask API on `http://127.0.0.1:5000`)*

**Terminal 2 (Frontend Dev Server):**
```powershell
cd frontend
npm run dev
```
*(Runs Vite Dev Server on `http://localhost:5173` with automatic API proxying to port 5000)*  
**Access URL:** Open your browser and navigate to:  
👉 **`http://localhost:5173`**

---

## 4. User Interface Walkthrough & Page Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION NAVBAR                        │
│ [Menu] EV Network Overview    [API & Models Ready] [83 Forecasts]      │
├───────────────┬────────────────────────────────────────────────────────┤
│ SIDEBAR       │ MAIN CONTENT AREA                                      │
│ ───────────── │ ────────────────────────────────────────────────────── │
│ 📊 Overview   │ [Methodological & Capacity Notice Banner]              │
│ 📈 Forecasts  │                                                        │
│ 📍 Stations   │ [KPI Grid: Total Sessions | Energy | Sites | Peaks]    │
│ 💡 Suggestions│                                                        │
│ 📉 Evaluation │ [Interactive Recharts Charts & Tables]                 │
│               │                                                        │
└───────────────┴────────────────────────────────────────────────────────┘
```

### 1. Network Overview (Dashboard)
- **Top Metric Cards:** Displays verified historical totals: **259,411 sessions**, **2,216.49 MWh delivered energy**, **46 charging stations**, and the peak arrival hour (**11:00 PT** with 12.15% of sessions) vs. lowest demand hour (**03:00 PT** with 0.25%).
- **Weekday vs. Weekend Arrival Curve:** Interactive Recharts area chart comparing hourly arrival intensity across weekdays (commuter surges at 08:00 and 11:00) vs. weekends (unimodal midday peak).
- **Day of Week Bar Chart:** Visualizes Monday–Friday commuter dominance vs. Saturday–Sunday drop.
- **Multi-Year Demand Trajectory:** Shows monthly growth from 2011 to 2019 and the sharp COVID-19 drop in April 2020.
- **Level 1 vs. Level 2 Breakdown:** Compares session shares (98.2% L2 vs. 1.8% L1), energy delivered (99.35% L2 vs. 0.65% L1), and dwell durations (157 min L2 vs. 303 min L1).
- **Top 10 Stations Table:** Ranked by total sessions with one-click drill-down navigation to station details.

### 2. Demand Forecasts (Predictions)
- **Station Selector:** Search and select any of the 46 stations, grouped cleanly into *Established Stations* vs. *Cold-Start Stations*.
- **Horizon Toggle:** Switch seamlessly between:
  - **1-Hour-Ahead:** Real-time Poisson arrival rate ($p_f$) for target hour $T_{\text{last}} + 1\text{h}$.
  - **24-Hour Day-Ahead:** 24 individual hourly step predictions for day-ahead operational planning.
- **Prediction Cards:** Displays expected arrivals (floating-point Poisson rate), rounded count ($\approx n$ sessions), station's historical same-hour average, and pre-hour concurrent plugged-in vehicles.
- **24-Hour Hourly Profile:** Visualizes predicted arrival bars for each upcoming clock hour alongside the station's historical hourly average curve.
- **Historical Backtest View:** Select 48h, 72h, or 168h to compare strictly causal model predictions against known historical arrivals with window MAE metrics.
- **Cold-Start Protection:** Selecting a station with $<24\text{h}$ of history displays a safe `insufficient_history` notification explaining that causal lag features require at least 24 hours of operating data.

### 3. Station Explorer
- **Search & Filter Directory:** Live search by station name, street address, or status filter (*All*, *Established*, *Cold-Start*).
- **Granular Telemetry:** Street address, geographic coordinates, operating dates, active days, dwell times, and port types.
- **Station Arrival Profiles:** Station-specific 24-hour mean arrival profile and recent active-day trends.
- **"Run Demand Forecast" Button:** Automatically pre-selects the station and redirects to the Predictions view.

### 4. Planning & Optimization Recommendations
- **Category Filter Chips:** Filter recommendations by *High Predicted Demand*, *Day-Ahead Planning*, *Recurring Peak Demand*, *Low Historical Demand*, or *Cold-Start Monitoring*.
- **Priority Filter:** Filter by *High*, *Medium*, *Low*, or *Info*.
- **Planning Cards:** Each recommendation displays the station name, priority badge, clear operational message, quantitative rationale ($1.4\times$ baseline, peak hours), metric tags, and direct forecast links.
- **Prominent Disclaimer:** Highlights that recommendations are non-prescriptive planning suggestions based on arrival rates and do not represent confirmed charger congestion.

### 5. Model Evaluation
- **Benchmark Tables:** Comprehensive comparison of XGBoost, Random Forest, Ridge Regression, and Baseline methods across Validation 2018, Test 2019, and OOD 2020.
- **Feature Importance:** Top 12 features ranked by permutation RMSE impact on held-out test data.
- **Test 2019 Calibration Curve:** Demonstrates near-perfect alignment between mean actual and mean predicted arrivals by clock hour.
- **Glossary:** Plain-language explanations of RMSE, MAE, count sparsity, and decimal arrival rates.

---

## 5. Troubleshooting & FAQ

### Issue: "Port 5000 is already in use"
**Cause:** Another process or previous Flask instance is still running on port 5000.  
**Resolution:**
```powershell
# Find process using port 5000 on Windows:
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
```

### Issue: "Vite build fails with EPERM error"
**Cause:** Transient Windows filesystem lock during directory creation.  
**Resolution:**
```powershell
cd frontend
npm run build
```
*(Running the command again resolves transient Windows file locks).*

### Issue: "Predictions return 'insufficient_history'"
**Explanation:** This is an intentional safety feature. 5 of the 13 Sherman Avenue stations opened in December 2020 and have only 1 to 8 recorded hours. The system safely withholds lag-based forecasts until 24 hours of telemetry are observed, preventing hallucinated predictions.

### Issue: "Browser shows blank page or outdated styles"
**Resolution:** Perform a hard refresh in your browser (**Ctrl + F5** on Windows or **Cmd + Shift + R** on Mac) to bypass local asset caching.
