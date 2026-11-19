# Project Screenshot Evidence Checklist
## Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations

**Project:** City of Palo Alto EV Charging Demand Prediction & Planning Decision Support System  
**Document:** Visual Evidence & Screenshot Capture Protocol  
**Version:** 1.0 (Final Submission)  
**Date:** November 2026  

---

## 1. Screenshot Capture Protocol

When capturing visual evidence for the project report, presentation slides, or university evaluation thesis:
1. Ensure the unified production server is actively running (`python backend/app.py`).
2. Use Google Chrome or Microsoft Edge at 100% zoom level (recommended resolution: 1920×1080).
3. Capture clean window screenshots without personal browser bookmarks or external toolbars.
4. Save images in PNG format inside an `evidence/` or `screenshots/` directory for inclusion in presentation decks.

---

## 2. Mandatory Screenshot Checklist (10 Items)

| # | Item Name | Navigation URL / Action | Key Visual Elements to Include | Verification Purpose |
| :-: | :--- | :--- | :--- | :--- |
| **01** | **Main Dashboard Overview** | `http://localhost:5000/` | - 4 Top KPI Stat Cards (259,411 sessions, 2,216.49 MWh, 46 Sites, 11:00 Peak)<br>- Top Banner Disclaimer<br>- Top 10 Stations Table | Proves live data integration and verified totals parity with cleaned CSV. |
| **02** | **Hourly Arrival Demand Chart** | `http://localhost:5000/` (Scroll) | - Weekday vs. Weekend Area Chart (Green weekday line, Blue weekend line)<br>- Hover tooltip showing arrival rates at 11:00 AM | Demonstrates temporal commuter surge vs. weekend demand collapse. |
| **03** | **Day of Week & Level Comparison** | `http://localhost:5000/` (Scroll) | - Day of Week Bar Chart (Mon–Fri green, Sat–Sun grey)<br>- Level 1 vs. Level 2 cards (98.2% L2 vs. 1.8% L1) | Shows port-type usage distribution and dwell time divergence. |
| **04** | **1-Hour-Ahead Demand Forecast** | `http://localhost:5000/predictions` | - Station: `PALO ALTO CA / HAMILTON #2`<br>- Horizon Toggle: `1-Hour-Ahead`<br>- Predicted session arrivals card (`0.25` arr/hr) vs. historical same-hour avg | Validates real-time high-frequency XGBoost inference pipeline. |
| **05** | **24-Hour Day-Ahead Demand Forecast** | `http://localhost:5000/predictions` | - Horizon Toggle: `24-Hour Day-Ahead`<br>- 24-Hour Forecast Bar Curve vs. Historical Line<br>- Historical Backtest chart below | Shows leak-free day-ahead forecasting and actual-vs-predicted backtest tracking. |
| **06** | **Station Explorer Directory & Telemetry** | `http://localhost:5000/stations` | - Station: `PALO ALTO CA / SHERMAN #1`<br>- Status Badge: `Insufficient History`<br>- Cold-start warning banner explaining $<24\text{h}$ of history | Demonstrates cold-start safety and refusal to hallucinate unobserved data. |
| **07** | **Optimization & Planning Recommendations** | `http://localhost:5000/recommendations` | - Category Filter Chips<br>- High Predicted Demand, Day-Ahead Top Demand, and Recurring Peak cards<br>- Top Methodological Disclaimer Banner | Confirms rule-based explainability and strict avoidance of unsupported congestion claims. |
| **08** | **Model Evaluation Benchmarks** | `http://localhost:5000/model-performance` | - Model Comparison Table with Val 2018 / Test 2019 / OOD 2020 tabs<br>- Permutation Feature Importance Bar Chart<br>- Test 2019 Calibration Curve | Documents chronological ML evaluation, RMSE/MAE metrics, and feature ranking. |
| **09** | **Backend Health & Model Readiness Endpoint** | `http://localhost:5000/api/health` | - Raw JSON output in browser/Postman:<br>`{"status": "healthy", "models": {"models_ready": true, "model_1h_loaded": true, "model_24h_loaded": true}}` | Proves backend service health, singleton model loading, and SQLite connection. |
| **10** | **Successful Server Startup & Automated Tests** | PowerShell Terminal Window | - Command: `python scripts/test_backend_api.py`<br>- Terminal output showing `ALL BACKEND TESTS PASSED!`<br>- Running `python backend/app.py` listening on port 5000 | Provides command-line terminal proof of testing completion and deployment. |

---

## 3. Recommended Filename Scheme

When saving screenshots for thesis submission:
1. `screenshot_01_dashboard_overview.png`
2. `screenshot_02_hourly_demand_curve.png`
3. `screenshot_03_day_of_week_and_levels.png`
4. `screenshot_04_prediction_1h_ahead.png`
5. `screenshot_05_prediction_24h_ahead.png`
6. `screenshot_06_station_explorer_cold_start.png`
7. `screenshot_07_planning_recommendations.png`
8. `screenshot_08_model_evaluation_benchmarks.png`
9. `screenshot_09_api_health_endpoint.png`
10. `screenshot_10_terminal_test_suite_pass.png`
