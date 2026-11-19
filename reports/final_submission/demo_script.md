# University Project Demonstration Script (5–7 Minutes)
## Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations

**Project:** City of Palo Alto EV Charging Demand Prediction & Planning Decision Support System  
**Document:** Live Demonstration & Presentation Script  
**Target Audience:** University Examination Committee, Evaluators & Industry Panel  
**Estimated Duration:** 6 minutes 30 seconds  

---

## Demonstration Timeline & Step-by-Step Script

```
00:00 ─── 01:00  [Step 1] Project Introduction & Problem Statement
01:00 ─── 01:45  [Step 2] Dataset Scope, Preprocessing & Ground Truth
01:45 ─── 02:30  [Step 3] Exploratory Data Analysis & Commuter Patterns
02:30 ─── 03:30  [Step 4] ML Methodology, Anti-Leakage & Model Selection
03:30 ─── 05:00  [Step 5] Live Demo: 1-Hour & 24-Hour Day-Ahead Demand Forecasts
05:00 ─── 05:45  [Step 6] Live Demo: Station Explorer & Cold-Start Safety
05:45 ─── 06:15  [Step 7] Live Demo: Explainable Planning Recommendations
06:15 ─── 06:45  [Step 8] Model Benchmarking & Honest Discussion of 2020 OOD Test
06:45 ─── 07:15  [Step 9] Summary, Limitations & Viva Q&A Readiness
```

---

### Step 1: Project Introduction & Problem Statement (00:00 – 01:00)
- **Screen View:** Project Title Screen / Application Homepage (`http://localhost:5000`)
- **Action:** Open application navbar showing *"EV Demand AI — Palo Alto Network"*.
- **Presenter Speaking Points:**
  > "Respected committee members, welcome to our presentation on **Electric Vehicle Charging Station Optimization: Predicting Demand for EV Stations**.
  >
  > As EV adoption accelerates, municipal charging networks face severe temporal demand imbalances. Charging stations frequently experience intense morning arrival surges followed by prolonged evening dormancy. Traditional fixed schedules fail to anticipate these fluctuations, while naive capacity expansion is capital-prohibitive.
  >
  > Our project bridges this gap by engineering a complete, end-to-end predictive decision support system using **10 years of public charging telemetry from the City of Palo Alto, California**. Our goal is to forecast station-level charging session arrival flow 1 hour and 24 hours ahead, enabling data-driven operational planning without data leakage."

---

### Step 2: Dataset Scope & Preprocessing Rigor (01:00 – 01:45)
- **Screen View:** Network Overview Dashboard (KPI Cards)
- **Action:** Point to the 4 KPI cards at the top of the dashboard.
- **Presenter Speaking Points:**
  > "The foundation of our project is the City of Palo Alto municipal charging dataset, spanning July 2011 through December 2020.
  >
  > Through systematic cleaning in Phase 3, we deduplicated records, resolved complex Excel timestamp anomalies, standardized timestamps into Pacific Time with accurate DST offsets, and recovered missing port types.
  >
  > As shown on our live dashboard, the verified cleaned dataset contains exactly **259,411 valid charging sessions**, delivering **2,216.49 MWh of cumulative energy** across **46 charging stations**."

---

### Step 3: Exploratory Discoveries & Commuter Dynamics (01:45 – 02:30)
- **Screen View:** Network Overview Dashboard (Hourly Arrival Curve & Level Comparison)
- **Action:** Scroll to the *Hourly Session Arrivals* and *Level 1 vs Level 2* charts.
- **Presenter Speaking Points:**
  > "Our exploratory analysis revealed critical operational patterns:
  >
  > First, charging demand is strongly commuter-driven. As seen in this green area chart, weekday session arrivals surge between 08:00 and 12:00 PT, peaking at 11:00 AM with over 31,000 cumulative sessions. In contrast, weekend demand drops by 45% and shifts to a flatter midday curve.
  >
  > Second, Level 2 commercial charging dominates the network, accounting for **98.2% of sessions and 99.35% of delivered energy**. Level 1 users dwell twice as long (over 5 hours) while drawing less than half the energy.
  >
  > Third, energy demand peaks two hours earlier (09:00 AM) than session count (11:00 AM) because early-morning commuters arrive with lower battery states of charge."

---

### Step 4: Machine Learning Methodology & Anti-Leakage (02:30 – 03:30)
- **Screen View:** Sidebar click to **Model Evaluation** page
- **Action:** Scroll to the *Anti-Leakage Split Design* section.
- **Presenter Speaking Points:**
  > "A primary contribution of our engineering work is strict chronological data splitting to prevent data leakage:
  > - We trained exclusively on historical data from 2011 to 2017.
  > - We tuned hyperparameters strictly on the 2018 validation set.
  > - We kept the entire year 2019 completely untouched until final testing.
  > - And we reserved the year 2020 as a separate out-of-distribution stress test due to the COVID-19 demand collapse and 13 new station commissioning.
  >
  > In Phase 4, we transformed session logs into a contiguous hourly panel of **1,533,050 station-hours**. We engineered 39 causal features including calendar harmonics, expanding station-hour means, concurrent plugged-in vehicle counts, and strictly backward-looking lags. Crucially, our 24-hour day-ahead model uses only features available at $t - 24\text{h}$, completely avoiding recursive synthetic predictions."

---

### Step 5: Live Forecasting Demonstration (03:30 – 05:00)
- **Screen View:** Sidebar click to **Demand Forecasts** page (`/predictions`)
- **Action 1:** Select station `PALO ALTO CA / HAMILTON #2` from the dropdown. Toggle between **1-Hour-Ahead** and **24-Hour Day-Ahead**.
- **Presenter Speaking Points:**
  > "Now let us demonstrate real-time model inference.
  >
  > Here, our pre-trained XGBoost model predicts that Hamilton #2 will experience **0.25 session arrivals** in the next clock hour (~0 to 1 arrivals), compared to its historical same-hour average of 0.28 arrivals.
  >
  > When we switch to the **24-Hour Day-Ahead forecast**, the model projects **6.96 total arrivals** over the next 24 hours, peaking at 10:00 AM. In the chart below, you can see the hour-by-hour forecast bars compared against the yellow historical baseline."
- **Action 2:** Scroll down to the **Historical Backtest** chart. Toggle the window from `Last 48h` to `Last 72h`.
- **Presenter Speaking Points:**
  > "To verify forecasting accuracy in real time, our application includes an automated Historical Backtest mode. Here, the system evaluates the model against known historical arrivals over the past 72 hours. Notice that the green 1-hour prediction line and purple 24-hour line closely track the actual grey arrival bars, achieving a low window MAE of **0.20**."

---

### Step 6: Station Explorer & Cold-Start Safety (05:00 – 05:45)
- **Screen View:** Sidebar click to **Station Explorer** (`/stations`)
- **Action:** Filter by *Cold-Start* and select `PALO ALTO CA / SHERMAN #1`.
- **Presenter Speaking Points:**
  > "Our Station Explorer provides full visibility into all 46 sites.
  >
  > Notice this critical safety feature: when we inspect Sherman #1, the system displays an **'Insufficient History' badge**.
  >
  > In December 2020, 13 new stations were commissioned on Sherman Avenue. Five of these stations have fewer than 24 hours of operating history. Rather than fabricating missing features or outputting misleading zero predictions, our system safely withholds lag-based forecasts until adequate telemetry is collected."

---

### Step 7: Explainable Planning Recommendations (05:45 – 06:15)
- **Screen View:** Sidebar click to **Optimization Plan** (`/recommendations`)
- **Action:** Highlight the top disclaimer banner and the category filter chips.
- **Presenter Speaking Points:**
  > "On our Optimization page, we translate machine learning forecasts into actionable municipal planning suggestions across 5 categories: High Predicted Demand, Day-Ahead Planning, Recurring Peak Demand, Low Historical Demand, and Cold-Start Monitoring.
  >
  > Notice the top banner: we maintain strict methodological discipline. Because municipal datasets lack physical port counts and queue lengths, our recommendations are explicitly labeled as **Planning Suggestions based on arrival flow**, avoiding ungrounded claims of hardware congestion."

---

### Step 8: Model Benchmarking & Honest Discussion of 2020 (06:15 – 06:45)
- **Screen View:** Sidebar click to **Model Evaluation** (`/model-performance`)
- **Action:** Switch table split tabs from `Val 2018` to `Main Test 2019` to `OOD 2020`.
- **Presenter Speaking Points:**
  > "In university research, transparency is paramount.
  >
  > On the held-out 2019 Test set, XGBoost decisively won across all primary metrics, achieving an RMSE of **0.3564** (a 7.3% error reduction over baseline) and an $R^2$ of **0.2723**.
  >
  > However, on the 2020 COVID-19 stress test, our evaluation honestly reports that **Random Forest achieved a slightly lower overall 1-hour RMSE (0.2661 vs. 0.2696)** because its shallower ensemble regularized better against the sudden shelter-in-place collapse. Conversely, XGBoost proved vastly superior on newly opened cold-start stations, achieving an $R^2$ of **+0.242** where Random Forest collapsed to **-0.197**."

---

### Step 9: Conclusion, Limitations & Viva Q&A (06:45 – 07:15)
- **Screen View:** Return to **Network Overview**
- **Presenter Speaking Points:**
  > "In conclusion, our project demonstrates a complete, leak-free machine learning system deployed in a unified web architecture.
  >
  > Our key limitations are the absence of physical port counts, real-time charger uptime status, and vehicle queue lengths. In future work, integrating real-time ChargePoint API telemetry and dynamic pricing modules would enable active load shifting.
  >
  > Thank you for your time. We are now ready to take your questions."

---

## Anticipated Viva & Evaluation Questions

#### Q1: "Why is your $R^2$ around 0.27 to 0.32? Isn't that low?"
- **Response:** "In hourly count forecasting at a single charging station, over **85% of hours have exactly zero arrivals**. When a vehicle does arrive, the count is an integer (1 or 2), whereas the expected Poisson arrival rate is a continuous fraction (e.g., 0.35). Due to the irreducible variance of individual driver arrival decisions ($\text{Var}(Y) = \lambda$), theoretical $R^2$ is bounded at the hourly resolution. What truly matters is our low RMSE (0.35), low MAE (0.20), and exceptional volume calibration bias of only -3.4%."

#### Q2: "How do you guarantee that your 24-hour forecast does not use future data?"
- **Response:** "For any target step $k \in [1, 24]$ at time $t_k$, all lags and rolling windows are constructed using data strictly $\le t_k - 24\text{h} \le T_{\text{last}}$. Lags are taken at 24h, 48h, 168h, and 336h. We do not use recursive multi-step forecasting where predicted values are fed back as lags, completely eliminating recursive error drift."

#### Q3: "Can your system tell if a station is currently full?"
- **Response:** "No, and we explicitly state this limitation. The City of Palo Alto dataset records session arrival events and energy draw, but does not include verified total port capacities, active out-of-order logs, or queue lengths. We predict arrival flow rate, which serves as a demand pressure indicator for planning."
