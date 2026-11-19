# Phase 9 Research Report: Calibrated Probabilistic Arrival-Count Forecasting & Distribution-Propagated Erlang-B Availability

**Project:** Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations  
**Location:** City of Palo Alto, California Municipal Charging Network  
**Dataset:** 259,344 clean charging sessions, 1,533,050 station-hours (2011–2020)  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary & Research Gap Addressed

Prior literature in EV charging demand forecasting and availability estimation suffers from a fundamental methodological divide:
1. Demand forecasting studies generate single **point estimates** ($\hat{y}$ expected arrivals/hour), obscuring the severe sparsity and zero-inflation of station-hour demand (where **84.67% of hours experience exactly zero arrivals**). A point prediction of $0.20$ arrivals/hour hides whether the charger is 80% likely to be empty or has a 4% risk of simultaneous arrivals.
2. Queueing-based availability models (such as Erlang-B loss models $B(c, a)$) assume constant, static offered traffic loads ($a = \lambda / \mu$), discarding the forecast uncertainty of arrival rates $\lambda$.
3. Because $B(c, a)$ is non-linear and convex in offered load $a$ over low-to-moderate traffic regimes, **Jensen’s inequality** dictates:
   $$\mathbb{E}[B(c, \lambda \cdot \text{dwell})] \ne B(c, \mathbb{E}[\lambda] \cdot \text{dwell})$$
   Feeding an uncalibrated point forecast into an Erlang model introduces systematic bias and produces false operational confidence.

**Phase 9 Contribution:**
We implemented and validated an end-to-end probabilistic decision pipeline:
$$\text{Feature Pipeline} \longrightarrow \text{Calibrated Count PMF } P(Y = k) \longrightarrow \text{Strict Proper Scoring Rules} \longrightarrow \text{Erlang-B Monte Carlo Uncertainty} \longrightarrow P(\text{All Ports Occupied}) \in [p_{05}, p_{95}]$$

On **6,804 ground-truth historical hours**, distribution-propagated Erlang-B reduced full-station occupancy Brier Score from **0.05204** (point baseline) to **0.03634**—a **+30.17% relative error reduction**—while providing calibrated 5–95% uncertainty intervals for driver decision support.

---

## 2. Chronological Data Splitting & Leakage Guards

To prevent look-ahead bias and temporal leakage, all models and preprocessors strictly adhere to the project's causal timeline:

| Split Segment | Time Span | Station-Hours | Stations | Purpose |
|---|---|---|---|---|
| **Training** | 2011–2017 | 707,668 | 27 historical | Model fitting & dispersion parameter estimation |
| **Validation** | 2018 | 277,536 | 27 historical + 6 new | Hyperparameter selection via strictly proper scoring rules |
| **Final Retrain** | 2011–2018 | 985,204 | 33 established | Final model fitting prior to blind evaluation |
| **Main Test** | 2019 | 289,080 | 33 historical | In-distribution out-of-sample evaluation |
| **COVID OOD Test**| 2020 | 258,766 | 33 hist + 13 Sherman | Out-of-distribution stress test (-87.1% demand drop) |

*Anti-leakage enforcement:* Preprocessors compute station encodings and imputation medians strictly on training sets ($t \le 2017$). Day-ahead features are restricted to information available strictly $\le t - 24\text{h}$.

---

## 3. Evaluated Model Architecture

Five discrete count models were trained and benchmarked:

1. **Poisson GLM (Baseline):**
   Log-linear link $\mu_i = \exp(X_i^\top \beta)$ with L2 regularization ($\alpha = 10^{-4}$). PMF:
   $$P(Y = k) = \frac{\mu_i^k e^{-\mu_i}}{k!}$$
2. **Negative Binomial (NB2 - Main):**
   Estimates dispersion $\alpha$ where $\text{Var}(Y) = \mu + \alpha \mu^2$ via maximum profile log-likelihood:
   $$P(Y = k) = \frac{\Gamma(k + 1/\alpha)}{\Gamma(k + 1)\Gamma(1/\alpha)} \left(\frac{1}{1 + \alpha \mu}\right)^{1/\alpha} \left(\frac{\alpha \mu}{1 + \alpha \mu}\right)^k$$
3. **Hurdle Negative Binomial (Sparse / Zero-Heavy):**
   Two-stage formulation:
   - Stage 1: Logistic regression for zero probability $\pi_0 = P(Y = 0 | X)$.
   - Stage 2: Zero-truncated Negative Binomial for positive counts $Y \ge 1$:
     $$P(Y = k) = (1 - \pi_0) \frac{P_{\text{NB}}(k)}{1 - P_{\text{NB}}(0)}, \quad k \ge 1$$
4. **XGBoost-as-Poisson:**
   High-capacity gradient boosted trees predicting Poisson arrival intensity $\lambda = \max(10^{-6}, \hat{y}_{\text{xgb}})$.
5. **Empirical Hour-of-Week Baseline:**
   Non-parametric empirical arrival count frequencies grouped by $(\text{station\_id}, \text{hour}, \text{is\_weekend})$.

---

## 4. Empirical Evaluation Results

All metrics below are computed from the complete test splits.

### A. 2019 Main Test Set Benchmark (289,080 station-hours)

| Model | Log Score (NLL) ↓ | Ranked Prob Score (RPS) ↓ | Zero Brier Score ↓ | Busy ROC-AUC ↑ | Busy PR-AUC ↑ | 90% Interval Coverage | ECE P(≥1) ↓ | Randomized PIT KS $D$ ↓ |
|---|---|---|---|---|---|---|---|---|
| **XGBoost-as-Poisson** | **0.3523** | **0.1099** | **0.0940** | **0.8693** | **0.5096** | 99.10% | **0.0172** | **0.0144** |
| **Poisson GLM** | 0.3770 | 0.1180 | 0.1017 | 0.8366 | 0.4490 | 99.05% | 0.0294 | 0.0163 |
| **Negative Binomial** | 0.3771 | 0.1180 | 0.1017 | 0.8366 | 0.4490 | 99.05% | 0.0294 | 0.0163 |
| **Hurdle NB** | 0.4152 | 0.1211 | 0.1001 | 0.8394 | 0.4518 | 99.25% | **0.0150** | 0.0244 |
| **Empirical Baseline** | 0.4125 | 0.1290 | 0.1118 | 0.8012 | 0.3269 | 98.85% | 0.0490 | 0.0441 |

*Key Findings:*
- **Strict Proper Scoring:** XGBoost-as-Poisson achieved the lowest Log Score (0.3523) and RPS (0.1099), outperforming the empirical baseline by **14.6% in Log Score** and **14.8% in RPS**.
- **Overdispersion Finding:** In Palo Alto municipal telemetry, dispersion parameter $\alpha$ converged to $\approx 0.0009$. This indicates that conditioned on 39 granular temporal, spatial, and lag features, conditional arrival variance closely matches conditional mean ($\text{Var} \approx \mu$).
- **Calibration & PIT Uniformity:** Randomized PIT Kolmogorov-Smirnov test statistic was minimal for XGBoost-as-Poisson ($D = 0.0144$), confirming near-uniform calibration on discrete test counts.

---

### B. 2020 COVID Out-of-Distribution Stress Test (258,766 station-hours)

| Model | Log Score (NLL) ↓ | Ranked Prob Score (RPS) ↓ | Zero Brier Score ↓ | 90% Interval Coverage | ECE P(≥1) ↓ |
|---|---|---|---|---|---|
| **XGBoost-as-Poisson** | **0.2241** | **0.0633** | **0.0573** | 99.53% | **0.0332** |
| **Poisson GLM** | 0.2281 | 0.0638 | 0.0578 | 99.42% | **0.0232** |
| **Negative Binomial** | 0.2281 | 0.0638 | 0.0578 | 99.42% | **0.0232** |
| **Hurdle NB** | 0.2495 | 0.0676 | 0.0583 | 99.40% | 0.0225 |
| **Empirical Baseline** | 0.3287 | 0.0981 | 0.0909 | 99.53% | 0.1295 |

*Key Findings:*
- During the pandemic demand collapse (-87.1% sessions), the **empirical hour-of-week baseline experienced severe calibration collapse** (ECE degraded to **12.95%**; Brier worsened to 0.0909).
- ML models with dynamic lag inputs quickly adapted to zero-dominated pandemic hours, maintaining tight calibration ($\text{ECE} \le 3.32\%$).

---

## 5. Erlang-B Propagation with Uncertainty

### Mathematical Mechanics
Given discrete arrival PMF $P(Y = k)$, Monte Carlo sampling draws $S = 500$ realizations $\lambda^{(s)}$. For active hardware inventory capacity $c$:
1. Offered load: $a^{(s)} = \lambda^{(s)} \cdot \bar{D}$ where $\bar{D}$ is the empirical station plugged-in dwell hours.
2. Erlang-B recurrence:
   $$\frac{1}{B(k, a)} = 1 + \frac{k}{a} \cdot \frac{1}{B(k-1, a)}$$
3. Existing vehicles carryover:
   $$N_{\text{carry}} = \min(N_{\text{prior}} \cdot e^{-1/\bar{D}}, c)$$
4. Full-station blocking:
   $$P_{\text{all\_occupied}}^{(s)} = \min\left(1.0, \max\left(B(c, a^{(s)}), \left(\frac{N_{\text{carry}} + a^{(s)}(1 - B)}{c}\right)^c\right)\right)$$

### Operational Validation on Real Ground Truth (6,804 Station-Hours)
Evaluated on `historical_availability_recent.csv` (221 observed full-station events; 3.25% prevalence):

| Metric | Point Erlang-B Baseline | Distribution-Propagated Erlang-B | Delta / Improvement |
|---|---|---|---|
| **Brier Score (Full-Station)** | **0.05204** | **0.03634** | **+30.17% Error Reduction** |
| **Mean 90% Confidence Interval Width** | N/A (Point value) | **0.5314** | Calibrated decision bounds |
| **Full-Station Hours Evaluated** | 221 | 221 | Ground truth preserved |

*Jensen's Effect in Practice:* In hours with positive arrival volatility, the distribution expectation $\mathbb{E}[B(c, \lambda)]$ exceeds the point forecast $B(c, \mathbb{E}[\lambda])$, correctly elevating the estimated risk of port contention when small demand bursts occur.

---

## 6. Generated Publication Figures

All figures were generated from raw predictions and saved to `reports/probabilistic_evaluation/`:
1. `pit_histogram.png`: Randomized PIT uniformity check across 10 probability bins.
2. `reliability_diagram_ge1_ge2.png`: Empirical vs forecast calibration curves for $P(\text{arrivals} \ge 1)$ and $P(\text{arrivals} \ge 2)$.
3. `scoring_rules_comparison.png`: Bar chart of mean Log Score and RPS across all model families.
4. `zero_hour_calibration.png`: ROC-AUC and PR-AUC for idle ($k=0$) vs busy ($k \ge 1$) discrimination.
5. `arrival_pmf_examples.png`: Sample discrete PMFs contrasting an idle morning station against a commuter peak surge.

---

## 7. Data Limitations & Honest Disclosures

1. **Memoryless Exponential Dwell Assumption:**
   Erlang-B analytically assumes exponential service duration ($CV = 1.0$). Empirical EV charging dwells exhibit $CV = 0.8653$ (under-dispersed due to parking enforcement and battery capacity limits).
2. **Unrecorded Physical Queues:**
   ChargePoint session records log vehicle plug-in events, not physical queues of cars waiting in travel lanes or parking stalls.
3. **Hardware Telemetry Gaps:**
   Observed port counts reflect distinct active MAC/EVSE telemetry. Unrecorded hardware outages or ICE-ing (internal combustion vehicles blocking EV spots) cannot be captured from session logs.
