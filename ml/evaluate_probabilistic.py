"""
ml/evaluate_probabilistic.py
Rigorous offline evaluation and calibration verification for probabilistic arrival models.
Evaluates on:
- 2019 Main Test Set (289,080 station-hours; 33 stations)
- 2020 Out-of-Distribution COVID Stress Test (258,766 station-hours; 33 historical + 13 new)

Evaluates:
1. Poisson GLM
2. Negative Binomial (NB2)
3. Hurdle Negative Binomial
4. XGBoost-as-Poisson baseline (using saved Phase 5 XGBoost point predictions)
5. Empirical station-hour-of-week frequency baseline

Metrics computed:
- Proper scoring rules: Log Score (negative log-likelihood) and Ranked Probability Score (RPS)
- Randomized PIT (Probability Integral Transform) histogram and uniformity KS-test
- Reliability diagrams / calibration curves for P(arrivals >= 1) and P(arrivals >= 2) with ECE
- Central 90% prediction interval coverage and mean interval width
- Zero-hour handling: Zero Brier score, ROC-AUC, and PR-AUC for idle vs busy
- Per-station and overall slices
- Generates JSON summary, tabular CSVs, and publication figures in reports/probabilistic_evaluation/.
"""

import json
import os
import time
from typing import Dict, List, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common import (
    FEATURES_1H,
    FEATURES_CSV,
    MODELS_DIR,
    TARGET_COL,
    TabularPreprocessor,
    load_hourly_features,
)
from ml.probabilistic_arrivals import (
    EmpiricalFrequencyBaseline,
    HurdleNegativeBinomialModel,
    NegativeBinomialModel,
    PoissonGLMModel,
    XGBoostPoissonBaseline,
    compute_pmf_metrics,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "probabilistic_evaluation")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_COLORS = {
    "poisson_glm": "#2A9D8F",
    "negative_binomial": "#1F6F8B",
    "hurdle_nb": "#E76F51",
    "xgboost_as_poisson": "#457B9D",
    "empirical_baseline": "#8D99AE",
}
MODEL_LABELS = {
    "poisson_glm": "Poisson GLM",
    "negative_binomial": "Negative Binomial (NB2)",
    "hurdle_nb": "Hurdle Negative Binomial",
    "xgboost_as_poisson": "XGBoost-as-Poisson",
    "empirical_baseline": "Empirical Hour-of-Week",
}


def compute_randomized_pit(pmf: np.ndarray, y_true: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, float, float]:
    """
    Computes randomized Probability Integral Transform (PIT) for discrete count data:
    u_i = F(y_i - 1) + V_i * (F(y_i) - F(y_i - 1)) where V_i ~ Uniform(0, 1).
    Under perfect calibration, u_i ~ Uniform(0, 1).
    """
    rng = np.random.RandomState(seed)
    y_clip = np.clip(y_true.astype(np.int32), 0, pmf.shape[1] - 1)
    cdf = np.cumsum(pmf, axis=1)

    cdf_y = np.take_along_axis(cdf, y_clip[:, None], axis=1).ravel()
    cdf_prev = np.where(
        y_clip > 0,
        np.take_along_axis(cdf, (y_clip - 1)[:, None], axis=1).ravel(),
        0.0,
    )
    v = rng.uniform(0.0, 1.0, size=len(y_clip))
    u = cdf_prev + v * (cdf_y - cdf_prev)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    # Subsample 20k points for fast, robust KS uniformity test
    ks_sample = rng.choice(u, size=min(20000, len(u)), replace=False)
    ks_stat, ks_p = stats.kstest(ks_sample, "uniform")
    return u, float(ks_stat), float(ks_p)


def compute_calibration_curve(probs: np.ndarray, y_bin: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Computes reliability curve points and Expected Calibration Error (ECE) for binary threshold event.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = []
    bin_observed = []
    bin_predicted = []
    bin_counts = []
    ece = 0.0
    total_n = len(probs)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            pred_mean = float(np.mean(probs[mask]))
            obs_mean = float(np.mean(y_bin[mask]))
            bin_centers.append(float(0.5 * (bins[b] + bins[b + 1])))
            bin_predicted.append(round(pred_mean, 4))
            bin_observed.append(round(obs_mean, 4))
            bin_counts.append(count)
            ece += (count / total_n) * abs(pred_mean - obs_mean)

    return {
        "bin_centers": bin_centers,
        "bin_predicted": bin_predicted,
        "bin_observed": bin_observed,
        "bin_counts": bin_counts,
        "ece": round(float(ece), 4),
    }


def evaluate_pmf_predictions(pmf: np.ndarray, y_true: np.ndarray, max_k: int = 12) -> Dict:
    """Comprehensive scoring rule and calibration evaluation on a split."""
    y_true = np.asarray(y_true, dtype=np.int32)
    n = len(y_true)
    k_clip = np.clip(y_true, 0, max_k)

    # 1. Log Score (negative log likelihood)
    probs_true = np.take_along_axis(pmf, k_clip[:, None], axis=1).ravel()
    probs_true = np.clip(probs_true, 1e-12, 1.0)
    neg_log_score = float(-np.mean(np.log(probs_true)))

    # 2. Ranked Probability Score (RPS)
    cdf = np.cumsum(pmf, axis=1)
    indicators = (k_clip[:, None] <= np.arange(max_k + 1)[None, :]).astype(np.float64)
    rps = float(np.mean(np.sum((cdf - indicators) ** 2, axis=1)))

    # 3. Prediction Intervals (90% and 50%)
    q05 = np.argmax(cdf >= 0.05, axis=1)
    q95 = np.argmax(cdf >= 0.95, axis=1)
    cov_90 = float(np.mean((k_clip >= q05) & (k_clip <= q95)))
    width_90 = float(np.mean(q95 - q05))

    q25 = np.argmax(cdf >= 0.25, axis=1)
    q75 = np.argmax(cdf >= 0.75, axis=1)
    cov_50 = float(np.mean((k_clip >= q25) & (k_clip <= q75)))
    width_50 = float(np.mean(q75 - q25))

    # 4. Zero arrivals and binary classification (Idle vs Busy)
    p_zero = pmf[:, 0]
    is_zero = (y_true == 0).astype(np.float64)
    zero_brier = float(np.mean((p_zero - is_zero) ** 2))

    p_busy = 1.0 - p_zero
    is_busy = (y_true > 0).astype(np.float64)
    try:
        roc_auc_busy = float(roc_auc_score(is_busy, p_busy))
        pr_auc_busy = float(average_precision_score(is_busy, p_busy))
    except Exception:
        roc_auc_busy = 0.5
        pr_auc_busy = float(is_busy.mean())

    # 5. Reliability curves
    cal_ge_1 = compute_calibration_curve(p_busy, is_busy, n_bins=10)
    p_ge_2 = np.sum(pmf[:, 2:], axis=1)
    is_ge_2 = (y_true >= 2).astype(np.float64)
    cal_ge_2 = compute_calibration_curve(p_ge_2, is_ge_2, n_bins=10)

    # 6. Randomized PIT
    _, pit_ks_stat, pit_ks_p = compute_randomized_pit(pmf, y_true, seed=42)

    return {
        "n_samples": int(n),
        "log_score": round(neg_log_score, 4),
        "rps": round(rps, 4),
        "zero_brier": round(zero_brier, 4),
        "roc_auc_busy": round(roc_auc_busy, 4),
        "pr_auc_busy": round(pr_auc_busy, 4),
        "coverage_90": round(cov_90, 4),
        "mean_width_90": round(width_90, 4),
        "coverage_50": round(cov_50, 4),
        "mean_width_50": round(width_50, 4),
        "pit_ks_stat": round(pit_ks_stat, 4),
        "pit_ks_p": round(pit_ks_p, 4),
        "ece_ge_1": cal_ge_1["ece"],
        "ece_ge_2": cal_ge_2["ece"],
        "cal_ge_1": cal_ge_1,
        "cal_ge_2": cal_ge_2,
    }


def generate_publication_figures(results: Dict, test_pmfs: Dict, y_test: np.ndarray):
    """Generates 5 publication-ready PNG figures in reports/probabilistic_evaluation/."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Randomized PIT Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    models_to_plot = ["poisson_glm", "negative_binomial", "hurdle_nb"]
    rng = np.random.RandomState(42)

    for idx, m_key in enumerate(models_to_plot):
        ax = axes[idx]
        pmf = test_pmfs[m_key]
        u, ks_stat, ks_p = compute_randomized_pit(pmf, y_test, seed=42)
        ax.hist(u, bins=10, range=(0, 1), density=True, color=MODEL_COLORS[m_key], alpha=0.75, edgecolor="white")
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform(0,1) Ref")
        ax.set_title(f"{MODEL_LABELS[m_key]}\nKS D={ks_stat:.3f} (p={ks_p:.1e})", fontsize=11, fontweight="bold")
        ax.set_xlabel("PIT Value u", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(loc="lower right", fontsize=9)
    plt.suptitle("Figure 1: Randomized PIT Uniformity on 2019 Main Test Set", fontsize=13, fontweight="bold", y=1.03)
    f1_path = os.path.join(REPORTS_DIR, "pit_histogram.png")
    plt.savefig(f1_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2: Reliability Diagrams P(arrivals >= 1) and P(arrivals >= 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")

    for m_key in ["poisson_glm", "hurdle_nb", "xgboost_as_poisson", "empirical_baseline"]:
        m_data = results["test_2019"][m_key]
        c1 = m_data["cal_ge_1"]
        c2 = m_data["cal_ge_2"]
        ax1.plot(c1["bin_predicted"], c1["bin_observed"], "o-", color=MODEL_COLORS[m_key], label=f"{MODEL_LABELS[m_key]} (ECE={c1['ece']:.3f})")
        ax2.plot(c2["bin_predicted"], c2["bin_observed"], "s-", color=MODEL_COLORS[m_key], label=f"{MODEL_LABELS[m_key]} (ECE={c2['ece']:.3f})")

    ax1.set_title("P(Arrivals >= 1) Reliability Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Forecast Probability", fontsize=10)
    ax1.set_ylabel("Empirical Frequency", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.set_title("P(Arrivals >= 2) Reliability Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Forecast Probability", fontsize=10)
    ax2.set_ylabel("Empirical Frequency", fontsize=10)
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(0, 0.6)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("Figure 2: Reliability Diagrams on 2019 Main Test Set", fontsize=13, fontweight="bold", y=1.02)
    f2_path = os.path.join(REPORTS_DIR, "reliability_diagram_ge1_ge2.png")
    plt.savefig(f2_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 3: Scoring Rules Comparison Bar Chart (Log Score & RPS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    model_order = ["empirical_baseline", "xgboost_as_poisson", "poisson_glm", "negative_binomial", "hurdle_nb"]
    labels = [MODEL_LABELS[k] for k in model_order]
    colors = [MODEL_COLORS[k] for k in model_order]

    log_scores_19 = [results["test_2019"][k]["log_score"] for k in model_order]
    rps_scores_19 = [results["test_2019"][k]["rps"] for k in model_order]

    axes[0].barh(labels, log_scores_19, color=colors, alpha=0.85, edgecolor="#333333")
    axes[0].set_title("Mean Negative Log Score (Lower = Better)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Negative Log Likelihood", fontsize=10)
    for i, v in enumerate(log_scores_19):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    axes[1].barh(labels, rps_scores_19, color=colors, alpha=0.85, edgecolor="#333333")
    axes[1].set_title("Ranked Probability Score (RPS) (Lower = Better)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean RPS", fontsize=10)
    for i, v in enumerate(rps_scores_19):
        axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    plt.suptitle("Figure 3: Strictly Proper Scoring Rule Comparison (2019 Test)", fontsize=13, fontweight="bold", y=1.03)
    f3_path = os.path.join(REPORTS_DIR, "scoring_rules_comparison.png")
    plt.savefig(f3_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 4: Zero-Hour Calibration (Idle vs Busy ROC & PR)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    roc_bars = [results["test_2019"][k]["roc_auc_busy"] for k in model_order]
    pr_bars = [results["test_2019"][k]["pr_auc_busy"] for k in model_order]

    axes[0].bar(labels, roc_bars, color=colors, alpha=0.85, edgecolor="#333333")
    axes[0].set_title("Busy-Hour ROC-AUC (Higher = Better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC", fontsize=10)
    axes[0].set_ylim(0.5, 0.95)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, pr_bars, color=colors, alpha=0.85, edgecolor="#333333")
    axes[1].set_title("Busy-Hour PR-AUC (Higher = Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("PR-AUC", fontsize=10)
    axes[1].set_ylim(0.1, 0.7)
    axes[1].tick_params(axis="x", rotation=25)

    plt.suptitle("Figure 4: Station Idle vs Busy Discrimination (2019 Test)", fontsize=13, fontweight="bold", y=1.03)
    f4_path = os.path.join(REPORTS_DIR, "zero_hour_calibration.png")
    plt.savefig(f4_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 5: Sample Arrival PMFs for Idle vs Busy station-hours
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sample_idle_pmf = test_pmfs["hurdle_nb"][0]
    # Find a busy hour
    busy_idx = int(np.argmax(y_test))
    sample_busy_pmf = test_pmfs["hurdle_nb"][busy_idx]
    k_vals = np.arange(len(sample_idle_pmf))

    axes[0].bar(k_vals[:7], sample_idle_pmf[:7], color="#2A9D8F", alpha=0.85, edgecolor="#1B4965")
    axes[0].set_title(f"Predicted PMF: Idle Station-Hour (True={y_test[0]})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Arrival Count k", fontsize=10)
    axes[0].set_ylabel("Probability P(Y = k)", fontsize=10)
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(k_vals[:7], sample_busy_pmf[:7], color="#E76F51", alpha=0.85, edgecolor="#6A040F")
    axes[1].set_title(f"Predicted PMF: Peak Congestion Hour (True={y_test[busy_idx]})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Arrival Count k", fontsize=10)
    axes[1].set_ylabel("Probability P(Y = k)", fontsize=10)

    plt.suptitle("Figure 5: Discrete Arrival Probability Mass Function (PMF) Examples", fontsize=13, fontweight="bold", y=1.03)
    f5_path = os.path.join(REPORTS_DIR, "arrival_pmf_examples.png")
    plt.savefig(f5_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70, flush=True)
    print("STARTING PROBABILISTIC CALIBRATION & PROPER SCORING EVALUATION", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    print("Loading hourly features...", flush=True)
    df_all = load_hourly_features(FEATURES_CSV)

    train_df = df_all[df_all["year"] <= 2017].reset_index(drop=True)
    test_df = df_all[df_all["year"] == 2019].reset_index(drop=True)
    ood_df = df_all[df_all["year"] == 2020].reset_index(drop=True)

    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int32)
    y_ood = ood_df[TARGET_COL].to_numpy(dtype=np.int32)

    print(f"Test 2019 rows: {len(test_df):,} | OOD 2020 rows: {len(ood_df):,}", flush=True)

    # Load trained models
    print("Loading models from models/ ...", flush=True)
    poisson_bundle = joblib.load(os.path.join(MODELS_DIR, "poisson_glm_1h.joblib"))
    nb_bundle = joblib.load(os.path.join(MODELS_DIR, "negative_binomial_1h.joblib"))
    hurdle_bundle = joblib.load(os.path.join(MODELS_DIR, "hurdle_nb_1h.joblib"))
    best_bundle = joblib.load(os.path.join(MODELS_DIR, "best_probabilistic_model_1h.joblib"))

    # Load Phase 5 XGBoost 1h model for XGBoost-as-Poisson baseline
    xgb_point_bundle = joblib.load(os.path.join(MODELS_DIR, "best_model_1h.joblib"))

    # Preprocess test and OOD splits
    prep = poisson_bundle["preprocessor"]
    X_test = prep.transform(test_df, impute=True)
    X_ood = prep.transform(ood_df, impute=True)

    # XGBoost preprocessing
    xgb_prep = xgb_point_bundle["preprocessor"]
    X_test_xgb = xgb_prep.transform(test_df, impute=False)
    X_ood_xgb = xgb_prep.transform(ood_df, impute=False)
    y_pred_xgb_test = xgb_point_bundle["model"].predict(X_test_xgb)
    y_pred_xgb_ood = xgb_point_bundle["model"].predict(X_ood_xgb)

    # 1. Empirical baseline
    emp_baseline = EmpiricalFrequencyBaseline(max_k=12).fit(train_df)
    test_pmf_emp = emp_baseline.predict_pmf(test_df)
    ood_pmf_emp = emp_baseline.predict_pmf(ood_df)

    # 2. Poisson GLM
    test_pmf_poisson = poisson_bundle["model"].predict_pmf(X_test, max_k=12)
    ood_pmf_poisson = poisson_bundle["model"].predict_pmf(X_ood, max_k=12)

    # 3. Negative Binomial
    test_pmf_nb = nb_bundle["model"].predict_pmf(X_test, max_k=12)
    ood_pmf_nb = nb_bundle["model"].predict_pmf(X_ood, max_k=12)

    # 4. Hurdle Negative Binomial
    test_pmf_hurdle = hurdle_bundle["model"].predict_pmf(X_test, max_k=12)
    ood_pmf_hurdle = hurdle_bundle["model"].predict_pmf(X_ood, max_k=12)

    # 5. XGBoost-as-Poisson
    xgb_poisson = XGBoostPoissonBaseline(xgb_point_bundle["model"])
    test_pmf_xgb = xgb_poisson.predict_pmf_from_preds(y_pred_xgb_test, max_k=12)
    ood_pmf_xgb = xgb_poisson.predict_pmf_from_preds(y_pred_xgb_ood, max_k=12)

    models_pmfs_test = {
        "poisson_glm": test_pmf_poisson,
        "negative_binomial": test_pmf_nb,
        "hurdle_nb": test_pmf_hurdle,
        "xgboost_as_poisson": test_pmf_xgb,
        "empirical_baseline": test_pmf_emp,
    }

    models_pmfs_ood = {
        "poisson_glm": ood_pmf_poisson,
        "negative_binomial": ood_pmf_nb,
        "hurdle_nb": ood_pmf_hurdle,
        "xgboost_as_poisson": ood_pmf_xgb,
        "empirical_baseline": ood_pmf_emp,
    }

    results = {"test_2019": {}, "ood_2020": {}, "by_station": {}}

    print("\n--- Evaluating Models on 2019 Main Test Set ---", flush=True)
    for m_key, pmf in models_pmfs_test.items():
        eval_dict = evaluate_pmf_predictions(pmf, y_test, max_k=12)
        results["test_2019"][m_key] = eval_dict
        print(
            f"[{m_key:20s}] LogScore: {eval_dict['log_score']:.4f} | RPS: {eval_dict['rps']:.4f} | "
            f"ZeroBrier: {eval_dict['zero_brier']:.4f} | 90%Cov: {eval_dict['coverage_90']:.4f} | "
            f"ROC-AUC: {eval_dict['roc_auc_busy']:.4f}",
            flush=True,
        )

    print("\n--- Evaluating Models on 2020 COVID Out-of-Distribution Split ---", flush=True)
    for m_key, pmf in models_pmfs_ood.items():
        eval_dict = evaluate_pmf_predictions(pmf, y_ood, max_k=12)
        results["ood_2020"][m_key] = eval_dict
        print(
            f"[{m_key:20s}] LogScore: {eval_dict['log_score']:.4f} | RPS: {eval_dict['rps']:.4f} | "
            f"ZeroBrier: {eval_dict['zero_brier']:.4f} | 90%Cov: {eval_dict['coverage_90']:.4f}",
            flush=True,
        )

    # Per-station slice analysis on 2019 Test for best model (poisson_glm / hurdle_nb)
    print("\n--- Computing Per-Station Slice Analysis ---", flush=True)
    best_pmf = test_pmf_poisson
    st_records = []
    for st_id, group in test_df.groupby("station_id"):
        st_idx = group.index.to_numpy() - test_df.index[0]
        st_y = y_test[st_idx]
        st_pmf = best_pmf[st_idx]
        st_eval = evaluate_pmf_predictions(st_pmf, st_y, max_k=12)
        st_records.append({
            "station_id": st_id,
            "n_hours": int(len(st_y)),
            "actual_arrivals": int(st_y.sum()),
            "arrival_rate": round(float(st_y.mean()), 3),
            "log_score": st_eval["log_score"],
            "rps": st_eval["rps"],
            "zero_brier": st_eval["zero_brier"],
            "coverage_90": st_eval["coverage_90"],
            "roc_auc_busy": st_eval["roc_auc_busy"],
        })
    results["by_station"] = st_records

    # Save CSV tables
    metrics_rows = []
    for split_tag, split_data in [("2019_test", results["test_2019"]), ("2020_ood", results["ood_2020"])]:
        for m_key, m_eval in split_data.items():
            metrics_rows.append({
                "split": split_tag,
                "model": m_key,
                "model_label": MODEL_LABELS[m_key],
                "log_score": m_eval["log_score"],
                "rps": m_eval["rps"],
                "zero_brier": m_eval["zero_brier"],
                "coverage_90": m_eval["coverage_90"],
                "mean_width_90": m_eval["mean_width_90"],
                "coverage_50": m_eval["coverage_50"],
                "roc_auc_busy": m_eval["roc_auc_busy"],
                "pr_auc_busy": m_eval["pr_auc_busy"],
                "ece_ge_1": m_eval["ece_ge_1"],
                "ece_ge_2": m_eval["ece_ge_2"],
                "pit_ks_stat": m_eval["pit_ks_stat"],
            })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(REPORTS_DIR, "metrics_by_model.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics table to {metrics_csv_path}", flush=True)

    st_df = pd.DataFrame(st_records)
    st_csv_path = os.path.join(REPORTS_DIR, "metrics_by_station.csv")
    st_df.to_csv(st_csv_path, index=False)
    print(f"Saved station breakdown to {st_csv_path}", flush=True)

    # Save JSON results
    json_path = os.path.join(REPORTS_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comprehensive evaluation JSON to {json_path}", flush=True)

    # Generate Figures
    print("\nGenerating publication figures in reports/probabilistic_evaluation/ ...", flush=True)
    generate_publication_figures(results, models_pmfs_test, y_test)
    print("All figures successfully saved!", flush=True)
    print(f"Evaluation complete in {time.time() - t0:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()
