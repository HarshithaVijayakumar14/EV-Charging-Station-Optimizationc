"""
ml/availability_calibration.py

Rigorous Empirical Availability Calibration & Error Propagation Engine.
Investigates the Research Gap:
"Existing studies forecast charging demand and, separately, station availability using queueing
or occupancy models. However, it remains unclear how arrival-forecast errors propagate through
dwell-time and queueing assumptions into port-availability estimates, and whether the resulting
full-station probabilities are calibrated under temporal distribution shift. This work evaluates
forecast-driven availability against occupancy reconstructed from session records, using calibration
and decision-level error metrics, and compares queueing-based estimates with empirical baselines."

CLI Usage:
    python -m ml.availability_calibration --horizon both --seed 42
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.common import FEATURES_1H, FEATURES_24H, MODELS_DIR
from ml.predict_demand import predict_from_dataframe
from ml.availability_prediction import availability_service, compute_erlang_b

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#D0D7DE",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "grid.alpha": 0.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CLEANED_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "ev_charging_cleaned.csv")
HOURLY_FEATURES_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "ev_charging_hourly_features.csv")
INVENTORY_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "charging_point_inventory.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "availability_calibration")


# ==============================================================================
# 1. Vectorized Erlang-B Loss Formula
# ==============================================================================

def vectorized_erlang_b(servers: np.ndarray, offered_load: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of Erlang-B blocking probability for c servers and offered traffic load a = lambda / mu.
    Uses stable backward recurrence:
        1 / B(k, a) = 1 + (k / a) * (1 / B(k-1, a))
    Supports heterogeneous servers and offered loads.
    Strictly bounded in [0.0, 1.0].
    """
    servers = np.asarray(servers, dtype=int)
    a = np.asarray(offered_load, dtype=float)
    out = np.zeros_like(a, dtype=float)

    max_c = int(np.max(servers)) if len(servers) > 0 else 0
    if max_c == 0:
        return np.ones_like(a)

    inv_b = np.ones_like(a)
    for k in range(1, max_c + 1):
        mask = (servers >= k) & (a > 1e-9)
        inv_b[mask] = 1.0 + (k / a[mask]) * inv_b[mask]
        save_mask = (servers == k) & (a > 1e-9)
        out[save_mask] = np.clip(1.0 / inv_b[save_mask], 0.0, 1.0)

    out[servers <= 0] = 1.0
    out[a <= 1e-9] = 0.0
    return np.clip(out, 0.0, 1.0)


# ==============================================================================
# 2. Capacity Mapping: Time-Varying vs Constant
# ==============================================================================

def build_capacity_lookups(inventory_csv: str = INVENTORY_CSV) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Builds:
    1. Time-varying active port intervals per station-port.
    2. End-of-data constant capacity map (historical_availability.py baseline).
    """
    inv_df = pd.read_csv(inventory_csv)
    inv_df["first_dt"] = pd.to_datetime(inv_df["first_observed"])
    inv_df["last_dt"] = pd.to_datetime(inv_df["last_observed"])

    # End-of-data constant capacity
    const_cap = inv_df.groupby("station_id")["is_active_end_of_data"].sum().to_dict()
    for st in inv_df["station_id"].unique():
        if const_cap.get(st, 0) == 0:
            const_cap[st] = 1

    return inv_df, const_cap


def assign_port_capacities(df: pd.DataFrame, inv_df: pd.DataFrame, const_cap: Dict[str, int]) -> pd.DataFrame:
    """
    Assigns both time-varying capacity C_tv and constant capacity C_const to every station-hour row.
    """
    df = df.copy()
    df["ts_dt"] = pd.to_datetime(df["hour_timestamp_pt"].str.slice(0, 19))

    # Constant capacity
    df["capacity_constant"] = df["station_id"].map(const_cap).fillna(2).astype(int)

    # Time-varying capacity: count how many ports in inv_df for station s have first_dt <= t and (last_dt >= t or is_active_end_of_data)
    # Optimized station-wise interval lookup
    st_intervals = {}
    for st, g in inv_df.groupby("station_id"):
        st_intervals[st] = [(r.first_dt, r.last_dt, r.is_active_end_of_data) for r in g.itertuples(index=False)]

    tv_caps = np.zeros(len(df), dtype=int)
    ts_arr = df["ts_dt"].values
    st_arr = df["station_id"].values

    # Station grouping for fast vectorized interval matching
    for st, intervals in st_intervals.items():
        st_mask = (st_arr == st)
        if not np.any(st_mask):
            continue
        st_times = ts_arr[st_mask]
        counts = np.zeros(len(st_times), dtype=int)
        for first_dt, last_dt, is_act_end in intervals:
            f_np = np.datetime64(first_dt)
            l_np = np.datetime64(last_dt)
            port_active = (st_times >= f_np) & ((st_times <= l_np) | (is_act_end and st_times >= f_np))
            counts += port_active.astype(int)
        # Ensure at least 1 port if station was active
        counts = np.maximum(counts, 1)
        tv_caps[st_mask] = counts

    # Fill any remaining zeros with constant capacity
    tv_caps[tv_caps == 0] = df["capacity_constant"].values[tv_caps == 0]
    df["capacity_time_varying"] = tv_caps
    df = df.drop(columns=["ts_dt"])
    return df


# ==============================================================================
# 3. Ground Truth Construction & Occupancy Definitions
# ==============================================================================

def compute_ground_truth(df: pd.DataFrame, cap_col: str = "capacity_time_varying") -> pd.DataFrame:
    """
    Reconstructs ground truth occupancy from session intervals.
    'Occupied' strictly means PLUGGED-IN (charging + idle duration occupying bay).
    Conservation: occupied + free + unknown == observed capacity.
    """
    df = df.copy()
    capacity = df[cap_col].values.astype(int)
    plugged = df["prior_plugged_in_vehicles_at_hour_start"].fillna(0).values.astype(int)

    # Occupied is bounded by observed capacity
    occupied = np.minimum(plugged, capacity)
    unknown = np.zeros_like(occupied, dtype=int)
    free = np.maximum(0, capacity - occupied - unknown)

    # Primary ground truth: full_event (occupied >= observed capacity)
    full_event = (occupied >= capacity).astype(int)

    df["observed_capacity"] = capacity
    df["true_occupied_points"] = occupied
    df["true_free_points"] = free
    df["true_full_event"] = full_event

    # Secondary ground truth: full at ANY time in the hour (concurrency >= capacity)
    # Upper bound: plugged + arrivals in hour
    arr = df["sessions_count"].fillna(0).values.astype(int)
    could_be_full = (plugged + arr) >= capacity
    # If plugged >= capacity, was full at start; if plugged + arr < capacity, never full
    df["true_full_anytime_hour"] = np.where(
        plugged >= capacity,
        1,
        np.where(could_be_full, 1, 0)  # conservative overlap bound
    ).astype(int)

    # Conservation assertion
    assert np.all(occupied + free + unknown == capacity), "Conservation identity violated!"
    return df


# ==============================================================================
# 4. Vectorized Model Predictions (A0, A1, A2, A3, B1, B2)
# ==============================================================================

def predict_availability_vectorized(
    df: pd.DataFrame,
    horizon: str = "1h",
    method: str = "A0",
    cap_col: str = "capacity_time_varying",
    empirical_rates: Optional[Dict] = None,
    empirical_survival: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of expected occupied points, expected free points, and P(zero free).
    Methods:
        A0: Current method (Erlang-B with forecast arrivals, carryover for 1h, none for 24h)
        A1: Same, but with TRUE arrivals (isolates queueing/dwell model error)
        A2: True arrivals + True hourly mean dwell (residual structural queueing error)
        A3: Empirical survival dwell model (non-exponential survival carryover)
        B1: Empirical baseline P(full | station, hour_of_week) with shrinkage
        B2: Persistence baseline (1h: state at hour start; 24h: state 24h prior)
    Returns:
        (exp_occupied, exp_free, prob_full)
    """
    obs_points = df[cap_col].values.astype(int)
    prior_plugged = df["prior_plugged_in_vehicles_at_hour_start"].fillna(0).values.astype(int)

    # 1. Arrival rates
    if method in ("A0", "A3"):
        arr_rate = np.clip(df[f"pred_arrivals_{horizon}"].values.astype(float), 0.0, None)
    elif method in ("A1", "A2"):
        arr_rate = np.clip(df["sessions_count"].values.astype(float), 0.0, None)
    else:
        arr_rate = np.clip(df[f"pred_arrivals_{horizon}"].values.astype(float), 0.0, None)

    # 2. Dwell times (strictly causal: station_hist_mean_plugged_dur_min)
    if method == "A2" and "true_hourly_mean_dwell_min" in df.columns:
        dwell_min = np.where(
            df["true_hourly_mean_dwell_min"].notna() & (df["true_hourly_mean_dwell_min"] > 0),
            df["true_hourly_mean_dwell_min"].values,
            df["station_hist_mean_plugged_dur_min"].fillna(157.0).values
        )
    else:
        dwell_min = df["station_hist_mean_plugged_dur_min"].fillna(157.0).values.astype(float)

    dwell_hours = np.maximum(dwell_min / 60.0, 0.5)
    service_rate = 1.0 / dwell_hours
    offered_load = arr_rate * dwell_hours

    # 3. Method B1: Empirical hour-of-week baseline
    if method == "B1":
        if empirical_rates is None:
            raise ValueError("B1 requires empirical_rates dictionary from training period.")
        keys = list(zip(df["station_id"], df["day_of_week"].astype(int) * 24 + df["hour"].astype(int)))
        prob_full = np.array([empirical_rates.get(k, 0.05) for k in keys], dtype=float)
        prob_full = np.clip(prob_full, 0.0, 1.0)
        exp_occupied = prob_full * obs_points.astype(float)
        exp_free = np.maximum(0.0, obs_points.astype(float) - exp_occupied)
        return exp_occupied, exp_free, prob_full

    # 4. Method B2: Persistence baseline
    if method == "B2":
        if horizon == "1h":
            # State at hour start
            prob_full = (prior_plugged >= obs_points).astype(float)
            exp_occupied = np.minimum(prior_plugged, obs_points).astype(float)
        else:
            # State 24h prior
            lag_occ = df.get("lag_24h_sessions", pd.Series(0, index=df.index)).fillna(0).values
            prob_full = np.clip((lag_occ >= 1.0).astype(float), 0.0, 1.0)
            exp_occupied = prob_full * obs_points.astype(float)
        exp_free = np.maximum(0.0, obs_points.astype(float) - exp_occupied)
        return exp_occupied, exp_free, prob_full

    # 5. Methods A0, A1, A2, A3 (Queueing / Erlang models)
    if horizon == "1h":
        if method == "A3" and empirical_survival is not None:
            # Non-exponential carryover probability S(1h) from empirical survival
            carryover_prob = float(empirical_survival[0])  # P(Dwell > 1 hour)
        else:
            carryover_prob = np.exp(-service_rate)
        expected_carryover = np.minimum(prior_plugged.astype(float) * carryover_prob, obs_points.astype(float))
    else:
        # 24h horizon assumes zero carryover
        expected_carryover = np.zeros_like(offered_load)

    # Erlang-B blocking probability
    p_blocked = vectorized_erlang_b(obs_points, offered_load)
    expected_new_occ = offered_load * (1.0 - p_blocked)

    exp_occ = np.minimum(expected_carryover + expected_new_occ, obs_points.astype(float))
    exp_occ = np.maximum(exp_occ, 0.0)
    est_free = np.maximum(0.0, obs_points.astype(float) - exp_occ)

    # Heuristic probability of zero free points from ml/availability_prediction.py
    # P_zero = min(1.0, max(p_blocked, (exp_occ / obs_points) ** obs_points))
    ratio = exp_occ / np.maximum(obs_points.astype(float), 1.0)
    poly = np.power(ratio, obs_points.astype(float))
    p_zero_free = np.clip(np.maximum(p_blocked, poly), 0.0, 1.0)

    return exp_occ, est_free, p_zero_free


# ==============================================================================
# 5. Calibration & Decision Evaluation Metrics
# ==============================================================================

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Computes calibration curve, ECE, MCE, Brier score decomposition, clipped Log Loss, AUROC, AUPRC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_true)

    if n == 0:
        return {}

    base_rate = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))

    # Quantile binning for reliability diagram
    # Handle ties and zero-heavy distributions via quantile edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(y_prob, quantiles))
    if len(bin_edges) <= 2:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    bin_data = []
    ece = 0.0
    mce = 0.0
    rel_sum = 0.0
    res_sum = 0.0

    for b in range(len(bin_edges) - 1):
        mask = (bin_indices == b)
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_prob = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        abs_err = abs(acc - mean_prob)
        weight = count / n
        ece += weight * abs_err
        mce = max(mce, abs_err)

        rel_sum += weight * ((mean_prob - acc) ** 2)
        res_sum += weight * ((acc - base_rate) ** 2)

        bin_data.append({
            "bin": int(b),
            "count": count,
            "mean_predicted_prob": round(mean_prob, 4),
            "empirical_accuracy": round(acc, 4),
            "calibration_error": round(abs_err, 4),
            "bin_lower": round(float(bin_edges[b]), 4),
            "bin_upper": round(float(bin_edges[b + 1]), 4),
        })

    # Brier decomposition: Uncertainty = y_bar * (1 - y_bar)
    uncertainty = base_rate * (1.0 - base_rate)
    reliability = rel_sum
    resolution = res_sum

    # Clipped Log Loss
    eps = 1e-6
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    ll = float(log_loss(y_true, y_prob_clipped))

    # Discrimination: AUROC and AUPRC
    try:
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        auroc = None

    try:
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        auprc = None

    return {
        "n_samples": n,
        "base_rate": round(base_rate, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "brier_reliability": round(reliability, 4),
        "brier_resolution": round(resolution, 4),
        "brier_uncertainty": round(uncertainty, 4),
        "log_loss": round(ll, 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "reliability_bins": bin_data,
    }


def compute_decision_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None,
    y_true_free: Optional[np.ndarray] = None,
    y_est_free: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes decision-level error metrics:
    1. Threshold sweep tau in [0.1, 0.9]: False Alarm Rate, Miss Rate, Precision, Recall, F1, Wasted Trip Rate.
    2. Continuous port availability error: MAE and RMSE of estimated free ports.
    3. 3-class availability confusion matrix: 0 free (full), 1 free (constrained), 2+ free (ample).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    threshold_results = []
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    for tau in thresholds:
        pred_full = (y_prob >= tau).astype(int)
        tp = int(np.sum((pred_full == 1) & (y_true == 1)))
        fp = int(np.sum((pred_full == 1) & (y_true == 0)))
        tn = int(np.sum((pred_full == 0) & (y_true == 0)))
        fn = int(np.sum((pred_full == 0) & (y_true == 1)))

        false_alarm_rate = fp / total_neg if total_neg > 0 else 0.0
        miss_rate = fn / total_pos if total_pos > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # Wasted trip rate: told free (pred_full == 0), but actually full (y_true == 1)
        wasted_trip_rate = fn / (tn + fn) if (tn + fn) > 0 else 0.0

        threshold_results.append({
            "threshold": round(tau, 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "false_alarm_rate": round(false_alarm_rate, 4),
            "miss_rate": round(miss_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "wasted_trip_rate": round(wasted_trip_rate, 4),
        })

    # Continuous free ports error metrics
    free_mae = None
    free_rmse = None
    three_class_cm = None

    if y_true_free is not None and y_est_free is not None:
        y_tf = np.asarray(y_true_free, dtype=float)
        y_ef = np.asarray(y_est_free, dtype=float)
        free_mae = round(float(mean_absolute_error(y_tf, y_ef)), 4)
        free_rmse = round(float(root_mean_squared_error(y_tf, y_ef)), 4)

        # 3-class classification:
        # 0: 0 free (full)
        # 1: 1 free (constrained)
        # 2: 2+ free (ample)
        true_class = np.where(y_tf <= 0.0, 0, np.where(y_tf <= 1.0, 1, 2))
        pred_class = np.where(y_ef < 0.5, 0, np.where(y_ef < 1.5, 1, 2))

        matrix = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = int(np.sum((true_class == i) & (pred_class == j)))

        total_samples = len(true_class)
        overall_acc = float(np.sum(true_class == pred_class)) / total_samples if total_samples > 0 else 0.0

        recalls = {}
        precisions = {}
        class_names = ["0_free", "1_free", "2_plus_free"]
        for c_idx, c_name in enumerate(class_names):
            c_true_total = int(np.sum(true_class == c_idx))
            c_pred_total = int(np.sum(pred_class == c_idx))
            tp_c = int(matrix[c_idx, c_idx])
            recalls[c_name] = round(tp_c / c_true_total, 4) if c_true_total > 0 else 0.0
            precisions[c_name] = round(tp_c / c_pred_total, 4) if c_pred_total > 0 else 0.0

        three_class_cm = {
            "classes": class_names,
            "matrix": matrix.tolist(),
            "overall_accuracy": round(overall_acc, 4),
            "per_class_recall": recalls,
            "per_class_precision": precisions,
        }

    return {
        "threshold_sweep": threshold_results,
        "free_ports_mae": free_mae,
        "free_ports_rmse": free_rmse,
        "three_class_confusion_matrix": three_class_cm,
    }


def fast_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast Mann-Whitney U based AUROC for bootstrap iterations (10x faster than sklearn)."""
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def block_bootstrap_confidence_intervals(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    cluster_cols: List[str] = ["station_id", "week"],
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% Confidence Intervals via block bootstrap over station-weeks (B=500, fixed seed).
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_cols].astype(str).agg("-".join, axis=1).values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Pre-index indices by cluster for fast sampling
    cluster_to_idx = {}
    for i, c in enumerate(clusters):
        cluster_to_idx.setdefault(c, []).append(i)
    cluster_to_idx = {c: np.array(idx, dtype=int) for c, idx in cluster_to_idx.items()}

    y_t = df[y_true_col].values.astype(int)
    y_p = df[y_prob_col].values.astype(float)

    brier_samples = []
    ece_samples = []
    auc_samples = []

    for _ in range(n_boot):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx_list = [cluster_to_idx[c] for c in sampled_c]
        b_idx = np.concatenate(idx_list)
        b_true = y_t[b_idx]
        b_prob = y_p[b_idx]

        brier_samples.append(float(np.mean((b_prob - b_true) ** 2)))
        ece_samples.append(float(np.mean(np.abs(b_prob - b_true))))
        auc_samples.append(fast_auc_score(b_true, b_prob))

    def ci_dict(samples):
        if len(samples) == 0:
            return {"mean": None, "ci_lower": None, "ci_upper": None}
        return {
            "mean": round(float(np.mean(samples)), 4),
            "ci_lower": round(float(np.percentile(samples, 2.5)), 4),
            "ci_upper": round(float(np.percentile(samples, 97.5)), 4),
        }

    return {
        "brier_score": ci_dict(brier_samples),
        "ece": ci_dict(ece_samples),
        "auroc": ci_dict(auc_samples),
    }


# ==============================================================================
# 6. Dwell Distribution Analysis & Goodness of Fit Test
# ==============================================================================

def analyze_dwell_distribution(cleaned_csv: str = CLEANED_CSV) -> Dict:
    """
    Tests whether Erlang-B's exponential service time assumption holds.
    Computes empirical mean, std, CV = sigma / mu, and Kolmogorov-Smirnov test against exponential.
    """
    cols = ["total_duration_min", "flag_duration_unreliable", "start_datetime_pt", "station_id"]
    df = pd.read_csv(cleaned_csv, usecols=cols, low_memory=False)
    df = df[~df["flag_duration_unreliable"].astype(bool)].copy()
    df["year"] = df["start_datetime_pt"].str.slice(0, 4).astype(int)

    splits = {
        "2011_2017_train": df[df["year"] <= 2017],
        "2018_val": df[df["year"] == 2018],
        "2019_test": df[df["year"] == 2019],
        "2020_ood": df[df["year"] == 2020],
        "overall": df,
    }

    results = {}
    for name, sub in splits.items():
        durations = sub["total_duration_min"].dropna().values
        n = len(durations)
        if n == 0:
            continue
        mean_d = float(np.mean(durations))
        std_d = float(np.std(durations))
        cv = std_d / mean_d if mean_d > 0 else 0.0

        # Subsample for KS test if huge to avoid memory spike
        sample_size = min(n, 10000)
        rng = np.random.default_rng(42)
        sample = rng.choice(durations, size=sample_size, replace=False)
        ks_res = stats.kstest(sample, "expon", args=(0, mean_d))

        results[name] = {
            "n_sessions": int(n),
            "mean_dwell_min": round(mean_d, 2),
            "std_dwell_min": round(std_d, 2),
            "coefficient_of_variation": round(cv, 4),
            "ks_statistic": round(float(ks_res.statistic), 4),
            "p_value": float(f"{ks_res.pvalue:.4e}"),
            "exponential_assumption_holds": bool(ks_res.pvalue >= 0.01 and 0.9 <= cv <= 1.1),
        }

    return results


# ==============================================================================
# 7. Distribution Shift: Population Stability Index (PSI) & 2-Sample KS
# ==============================================================================

def compute_distribution_shift(df_val: pd.DataFrame, df_test: pd.DataFrame, df_ood: pd.DataFrame) -> Dict:
    """
    Computes PSI and 2-Sample KS test statistics for arrival rates, dwell, and occupancy.
    """
    def psi(reference: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
        ref = reference[~np.isnan(reference)]
        tgt = target[~np.isnan(target)]
        if len(ref) == 0 or len(tgt) == 0:
            return 0.0
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.quantile(ref, quantiles))
        if len(bins) <= 2:
            bins = np.linspace(min(ref.min(), tgt.min()), max(ref.max(), tgt.max()), n_bins + 1)
        ref_counts = np.histogram(ref, bins=bins)[0]
        tgt_counts = np.histogram(tgt, bins=bins)[0]
        ref_pct = (ref_counts + 1e-4) / (len(ref) + 1e-4 * len(ref_counts))
        tgt_pct = (tgt_counts + 1e-4) / (len(tgt) + 1e-4 * len(tgt_counts))
        return float(np.sum((ref_pct - tgt_pct) * np.log(ref_pct / tgt_pct)))

    features = [
        ("arrivals_true", "sessions_count"),
        ("plugged_dur_causal", "station_hist_mean_plugged_dur_min"),
        ("occupancy_rate", "true_full_event"),
    ]

    out = {}
    for label, col in features:
        val_v = df_val[col].dropna().values
        test_v = df_test[col].dropna().values
        ood_v = df_ood[col].dropna().values

        # 2018 -> 2019
        psi_19 = psi(val_v, test_v)
        ks_19 = stats.ks_2samp(val_v[:5000], test_v[:5000])

        # 2018 -> 2020
        psi_20 = psi(val_v, ood_v)
        ks_20 = stats.ks_2samp(val_v[:5000], ood_v[:5000])

        out[label] = {
            "shift_2018_to_2019": {
                "psi": round(psi_19, 4),
                "ks_statistic": round(float(ks_19.statistic), 4),
                "p_value": float(f"{ks_19.pvalue:.4e}"),
                "shift_severity": "Moderate" if psi_19 > 0.1 else "Negligible",
            },
            "shift_2018_to_2020": {
                "psi": round(psi_20, 4),
                "ks_statistic": round(float(ks_20.statistic), 4),
                "p_value": float(f"{ks_20.pvalue:.4e}"),
                "shift_severity": "Severe" if psi_20 > 0.25 else ("Moderate" if psi_20 > 0.1 else "Negligible"),
            },
        }

    return out


# ==============================================================================
# 8. Error Propagation & Sensitivity Analysis
# ==============================================================================

def compute_error_propagation(df: pd.DataFrame, horizon: str = "1h") -> Dict:
    """
    Decomposes availability estimation error into:
    1. Arrival-forecast error: |A0 - A1|
    2. Dwell/queueing assumption error: |A1 - A2|
    3. Residual structural error: |A2 - True|
    """
    true_occ = df["true_occupied_points"].values.astype(float)
    a0_occ = df[f"{horizon}_A0_exp_occ"].values.astype(float)
    a1_occ = df[f"{horizon}_A1_exp_occ"].values.astype(float)
    a2_occ = df[f"{horizon}_A2_exp_occ"].values.astype(float)

    total_err = float(mean_absolute_error(true_occ, a0_occ))
    arr_err = float(mean_absolute_error(a1_occ, a0_occ))
    dwell_err = float(mean_absolute_error(a2_occ, a1_occ))
    residual_err = float(mean_absolute_error(true_occ, a2_occ))

    # Sensitivity: elasticity of expected occupancy and P(full) to +/- 10%, 25%, 50% arrival perturbations
    base_arr = df[f"pred_arrivals_{horizon}"].values.astype(float)
    obs_cap = df["capacity_time_varying"].values.astype(int)
    dwell_h = np.maximum(df["station_hist_mean_plugged_dur_min"].fillna(157.0).values / 60.0, 0.5)

    perturbations = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
    sensitivity_results = []
    base_occ = df[f"{horizon}_A0_exp_occ"].values
    base_pfull = df[f"{horizon}_A0_prob_full"].values

    for delta in perturbations:
        mod_arr = np.clip(base_arr * (1.0 + delta), 0.0, None)
        mod_load = mod_arr * dwell_h
        mod_pb = vectorized_erlang_b(obs_cap, mod_load)
        if horizon == "1h":
            prior = df["prior_plugged_in_vehicles_at_hour_start"].fillna(0).values.astype(float)
            co = np.minimum(prior * np.exp(-1.0 / dwell_h), obs_cap.astype(float))
            mod_occ = np.minimum(co + mod_load * (1.0 - mod_pb), obs_cap.astype(float))
        else:
            mod_occ = np.minimum(mod_load * (1.0 - mod_pb), obs_cap.astype(float))

        mod_pfull = np.clip(np.maximum(mod_pb, (mod_occ / np.maximum(obs_cap, 1)) ** obs_cap), 0.0, 1.0)

        mean_occ = float(np.mean(mod_occ))
        mean_pf = float(np.mean(mod_pfull))

        pct_d_occ = ((mean_occ - np.mean(base_occ)) / np.mean(base_occ) * 100.0) if np.mean(base_occ) > 0 else 0.0
        pct_d_pf = ((mean_pf - np.mean(base_pfull)) / np.mean(base_pfull) * 100.0) if np.mean(base_pfull) > 0 else 0.0

        sensitivity_results.append({
            "arrival_perturbation_pct": int(delta * 100),
            "mean_expected_occupied": round(mean_occ, 3),
            "pct_change_occupied": round(pct_d_occ, 2),
            "mean_prob_full": round(mean_pf, 4),
            "pct_change_prob_full": round(pct_d_pf, 2),
            "elasticity_prob_full": round(pct_d_pf / (delta * 100.0), 3) if delta != 0 else 1.0,
        })

    # Error binned by arrival forecast error magnitude |pred_arr - true_arr|
    arr_true = df["sessions_count"].fillna(0).values.astype(float)
    arr_abs_err = np.abs(base_arr - arr_true)
    bins = [0.0, 0.2, 0.5, 1.0, 2.0, np.inf]
    bin_labels = [
        "0.0-0.2 (Negligible)",
        "0.2-0.5 (Low)",
        "0.5-1.0 (Moderate)",
        "1.0-2.0 (High)",
        "2.0+ (Severe)",
    ]
    binned_errors = []
    p_full_a0 = df[f"{horizon}_A0_prob_full"].values
    y_full = df["true_full_event"].values

    for i in range(len(bins) - 1):
        mask = (arr_abs_err >= bins[i]) & (arr_abs_err < bins[i + 1])
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_arr_err = float(np.mean(arr_abs_err[mask]))
        mean_occ_err = float(mean_absolute_error(true_occ[mask], a0_occ[mask]))
        mean_res_err = float(mean_absolute_error(true_occ[mask], a2_occ[mask]))
        mean_pf_err = float(np.mean(np.abs(p_full_a0[mask] - y_full[mask])))
        full_freq = float(np.mean(y_full[mask]))
        binned_errors.append({
            "bin_range": bin_labels[i],
            "station_hour_count": count,
            "mean_arrival_error": round(mean_arr_err, 4),
            "mean_occupancy_mae": round(mean_occ_err, 4),
            "mean_prob_full_error": round(mean_pf_err, 4),
            "empirical_full_rate": round(full_freq, 4),
            "residual_structural_mae": round(mean_res_err, 4),
        })

    return {
        "total_mae_occupied": round(total_err, 4),
        "component_mae": {
            "arrival_forecast_error": round(arr_err, 4),
            "dwell_queueing_assumption_error": round(dwell_err, 4),
            "residual_structural_error": round(residual_err, 4),
        },
        "variance_contributions_pct": {
            "arrival_forecast_error_share": round((arr_err / (arr_err + dwell_err + residual_err)) * 100.0, 1) if (arr_err + dwell_err + residual_err) > 0 else 0,
            "dwell_queueing_assumption_error_share": round((dwell_err / (arr_err + dwell_err + residual_err)) * 100.0, 1) if (arr_err + dwell_err + residual_err) > 0 else 0,
            "residual_structural_error_share": round((residual_err / (arr_err + dwell_err + residual_err)) * 100.0, 1) if (arr_err + dwell_err + residual_err) > 0 else 0,
        },
        "sensitivity_elasticity": sensitivity_results,
        "error_binned_by_arrival_error": binned_errors,
    }


# ==============================================================================
# 9. Visualization & Plot Generation
# ==============================================================================

def generate_publication_figures(results: Dict, output_dir: str = RESULTS_DIR):
    """
    Renders and saves publication-quality PNG figures into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Reliability Diagrams by Year
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    years = [("2018_validation", "2018 Validation"), ("2019_test", "2019 Test"), ("2020_ood", "2020 OOD Stress Test")]
    for ax, (yr_key, yr_title) in zip(axes, years):
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        yr_data = results.get("evaluations", {}).get("1h", {}).get(yr_key, {})
        # A0
        a0_bins = yr_data.get("A0", {}).get("calibration", {}).get("reliability_bins", [])
        if a0_bins:
            p_a0 = [b["mean_predicted_prob"] for b in a0_bins]
            acc_a0 = [b["empirical_accuracy"] for b in a0_bins]
            ax.plot(p_a0, acc_a0, "s-", color="#1F6F8B", label="A0 Current (Erlang-B)", lw=2)
        # Recalibrated Isotonic C
        c_bins = yr_data.get("C_isotonic", {}).get("calibration", {}).get("reliability_bins", [])
        if c_bins:
            p_c = [b["mean_predicted_prob"] for b in c_bins]
            acc_c = [b["empirical_accuracy"] for b in c_bins]
            ax.plot(p_c, acc_c, "o-", color="#2A9D8F", label="C Recalibrated (Isotonic)", lw=2)
        # Empirical baseline B1
        b1_bins = yr_data.get("B1", {}).get("calibration", {}).get("reliability_bins", [])
        if b1_bins:
            p_b1 = [b["mean_predicted_prob"] for b in b1_bins]
            acc_b1 = [b["empirical_accuracy"] for b in b1_bins]
            ax.plot(p_b1, acc_b1, "^:", color="#E76F51", label="B1 Empirical Baseline", lw=1.5)

        ax.set_title(f"{yr_title} (1h Horizon)")
        ax.set_xlabel("Predicted Probability P(Full)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Empirical Frequency P(Full | B_m)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "reliability_diagram_by_year.png"))
    plt.close(fig)

    # 2. ECE and Brier Score by Year & Method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    methods = ["A0", "A1", "A3", "B1", "C_isotonic"]
    colors = ["#1F6F8B", "#457B9D", "#A8DADC", "#E76F51", "#2A9D8F"]
    x = np.arange(len(methods))
    width = 0.25

    for idx, (yr_key, yr_label) in enumerate(years):
        ece_vals = [results.get("evaluations", {}).get("1h", {}).get(yr_key, {}).get(m, {}).get("calibration", {}).get("ece", 0.0) for m in methods]
        brier_vals = [results.get("evaluations", {}).get("1h", {}).get(yr_key, {}).get(m, {}).get("calibration", {}).get("brier_score", 0.0) for m in methods]
        ax1.bar(x + idx * width, ece_vals, width, label=yr_label, alpha=0.85)
        ax2.bar(x + idx * width, brier_vals, width, label=yr_label, alpha=0.85)

    ax1.set_title("Expected Calibration Error (ECE, lower is better)")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("ECE")
    ax1.legend()

    ax2.set_title("Brier Score (lower is better)")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Brier Score")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ece_brier_by_year.png"))
    plt.close(fig)

    # 3. Error Decomposition Breakdown
    decomp = results.get("error_propagation", {}).get("1h", {}).get("2019_test", {}).get("variance_contributions_pct", {})
    if decomp:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = [
            "Arrival Forecast\nError (A0 - A1)",
            "Dwell & Queueing\nAssumption (A1 - A2)",
            "Residual Structural\nError (A2 - True)"
        ]
        shares = [
            decomp.get("arrival_forecast_error_share", 30),
            decomp.get("dwell_queueing_assumption_error_share", 40),
            decomp.get("residual_structural_error_share", 30),
        ]
        bars = ax.bar(labels, shares, color=["#E63946", "#F4A261", "#2A9D8F"], width=0.55)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Share of Availability Estimation Error (%)")
        ax.set_title("Error Propagation Decomposition (2019 Test Set, 1h Horizon)")
        ax.set_ylim(0, max(shares) + 12)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "error_decomposition.png"))
        plt.close(fig)

    # 4. Sensitivity & Elasticity Curves
    sens_data = results.get("error_propagation", {}).get("1h", {}).get("2019_test", {}).get("sensitivity_elasticity", [])
    if sens_data:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        deltas = [d["arrival_perturbation_pct"] for d in sens_data]
        pct_occ = [d["pct_change_occupied"] for d in sens_data]
        pct_pf = [d["pct_change_prob_full"] for d in sens_data]

        ax.plot(deltas, pct_occ, "o-", color="#1F6F8B", label="% Change in Expected Occupancy", lw=2)
        ax.plot(deltas, pct_pf, "s--", color="#E76F51", label="% Change in P(Zero Free Points)", lw=2)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Arrival Forecast Perturbation (%)")
        ax.set_ylabel("Response Change (%)")
        ax.set_title("Arrival-to-Availability Elasticity Response")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "sensitivity_elasticity.png"))
        plt.close(fig)

    # 5. Dwell vs Exponential Histogram
    dwell_stats = results.get("dwell_assumption_test", {})
    if dwell_stats:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        years_dwell = list(dwell_stats.keys())
        cvs = [dwell_stats[k]["coefficient_of_variation"] for k in years_dwell]
        labels = [k.replace("_", " ").title() for k in years_dwell]
        bars = ax.bar(labels, cvs, color="#457B9D", width=0.5)
        ax.axhline(1.0, color="#E63946", linestyle="--", lw=2, label="Exponential Assumption (CV = 1.0)")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"CV={yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Coefficient of Variation (sigma / mu)")
        ax.set_title("Plugged-In Dwell Time Dispersion vs. Exponential Distribution")
        ax.set_ylim(0, 1.25)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "dwell_vs_exponential.png"))
        plt.close(fig)

    # 6. Decision Curves (ROC & Precision-Recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    dec_data = results.get("decision_metrics", {}).get("1h", {}).get("2019_test", {}).get("A0", [])
    if isinstance(dec_data, dict):
        dec_data = dec_data.get("threshold_sweep", [])
    dec_c = results.get("decision_metrics", {}).get("1h", {}).get("2019_test", {}).get("C_isotonic", [])
    if isinstance(dec_c, dict):
        dec_c = dec_c.get("threshold_sweep", [])
    if dec_data and dec_c:
        far_a0 = [d["false_alarm_rate"] for d in dec_data]
        rec_a0 = [d["recall"] for d in dec_data]
        prec_a0 = [d["precision"] for d in dec_data]

        far_c = [d["false_alarm_rate"] for d in dec_c]
        rec_c = [d["recall"] for d in dec_c]
        prec_c = [d["precision"] for d in dec_c]

        # ROC: Recall (TPR) vs False Alarm Rate (FPR)
        ax1.plot(far_a0, rec_a0, "s-", color="#1F6F8B", label="A0 Current (Erlang-B)", lw=2)
        ax1.plot(far_c, rec_c, "o-", color="#2A9D8F", label="C Recalibrated", lw=2)
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("False Alarm Rate (P(Pred Full | Free))")
        ax1.set_ylabel("Recall / Hit Rate (P(Pred Full | Full))")
        ax1.set_title("Receiver Operating Characteristic (ROC)")
        ax1.legend()

        # PR: Precision vs Recall
        ax2.plot(rec_a0, prec_a0, "s-", color="#1F6F8B", label="A0 Current (Erlang-B)", lw=2)
        ax2.plot(rec_c, prec_c, "o-", color="#2A9D8F", label="C Recalibrated", lw=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Tradeoff")
        ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "decision_tradeoffs_roc_pr.png"))
    plt.close(fig)


# ==============================================================================
# 10. Main Pipeline Orchestrator
# ==============================================================================

def run_calibration_experiment(
    horizons: List[str] = ["1h", "24h"],
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
) -> Dict:
    """
    Executes the complete availability calibration, error propagation, and benchmark pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("STARTING AVAILABILITY CALIBRATION & ERROR PROPAGATION PIPELINE")
    print(f"Horizons: {horizons} | Random Seed: {seed}")
    print("=" * 80)

    # 1. Dwell distribution test
    print("\n1. Testing empirical dwell distribution vs. exponential assumption...")
    dwell_results = analyze_dwell_distribution(CLEANED_CSV)
    for k, v in dwell_results.items():
        print(f"   [{k}] N={v['n_sessions']}, Mean={v['mean_dwell_min']}m, CV={v['coefficient_of_variation']}, KS D={v['ks_statistic']} (p={v['p_value']})")

    # 2. Load hourly features for 2011-2020
    print("\n2. Loading hourly feature dataset and inventory...")
    inv_df, const_cap = build_capacity_lookups(INVENTORY_CSV)
    df_raw = pd.read_csv(HOURLY_FEATURES_CSV)

    # Separate cold-start stations (Sherman Ave opened in 2020)
    sherman_stations = sorted([s for s in df_raw["station_id"].unique() if "SHERMAN" in s])
    print(f"   Identified {len(sherman_stations)} cold-start Sherman stations (excluded from evaluation splits):")
    for s in sherman_stations[:4]:
        print(f"     - {s}")

    df_eval_all = df_raw[~df_raw["station_id"].isin(sherman_stations)].copy()
    df_eval_all = assign_port_capacities(df_eval_all, inv_df, const_cap)
    df_eval_all = compute_ground_truth(df_eval_all, cap_col="capacity_time_varying")

    # Add week column for block bootstrap
    df_eval_all["week"] = pd.to_datetime(df_eval_all["hour_timestamp_pt"].str.slice(0, 10)).dt.isocalendar().week

    # 3. Train empirical baseline B1 from pre-2018 data (2011-2017)
    print("\n3. Fitting empirical baseline B1 from pre-2018 historical data...")
    df_pre18 = df_eval_all[df_eval_all["year"] <= 2017].copy()
    df_pre18["hour_of_week"] = df_pre18["day_of_week"].astype(int) * 24 + df_pre18["hour"].astype(int)
    # Empirical Bayes shrinkage: alpha=10
    overall_mean = float(df_pre18["true_full_event"].mean())
    st_means = df_pre18.groupby("station_id")["true_full_event"].mean().to_dict()

    b1_rates = {}
    for (st, how), g in df_pre18.groupby(["station_id", "hour_of_week"]):
        k = (st, how)
        st_prior = st_means.get(st, overall_mean)
        shrunk_p = (g["true_full_event"].sum() + 10.0 * st_prior) / (len(g) + 10.0)
        b1_rates[k] = float(shrunk_p)

    # 4. Generate batch XGBoost arrival predictions for 2018-2020 evaluation splits
    print("\n4. Generating batch XGBoost demand forecasts for 2018-2020...", flush=True)
    df_eval = df_eval_all[df_eval_all["year"].isin([2018, 2019, 2020])].copy()
    for h in horizons:
        print(f"   Predicting {h}-ahead arrivals across 2018-2020 ({len(df_eval)} rows)...", flush=True)
        preds = predict_from_dataframe(df_eval, horizon=h)
        df_eval[f"pred_arrivals_{h}"] = preds

    # Empirical survival function for A3 from 2011-2017 sessions
    clean_sub = pd.read_csv(CLEANED_CSV, usecols=["total_duration_min", "start_datetime_pt", "flag_duration_unreliable"])
    clean_sub = clean_sub[~clean_sub["flag_duration_unreliable"].astype(bool) & (clean_sub["start_datetime_pt"].str.slice(0, 4).astype(int) <= 2017)]
    dur_arr = clean_sub["total_duration_min"].dropna().values
    emp_survival = np.array([np.mean(dur_arr > 60.0), np.mean(dur_arr > 120.0), np.mean(dur_arr > 180.0)])
    print(f"   Empirical survival P(Dwell > 1h)={emp_survival[0]:.3f}, P(Dwell > 2h)={emp_survival[1]:.3f}", flush=True)

    # Splits
    split_dfs = {
        "2018_validation": df_eval[df_eval["year"] == 2018].copy(),
        "2019_test": df_eval[df_eval["year"] == 2019].copy(),
        "2020_ood": df_eval[df_eval["year"] == 2020].copy(),
    }

    results = {
        "metadata": {
            "title": "EV Charging Station Availability Calibration & Error Propagation Analysis",
            "seed": seed,
            "horizons": horizons,
            "evaluation_splits": {
                "2018_validation": len(split_dfs["2018_validation"]),
                "2019_test": len(split_dfs["2019_test"]),
                "2020_ood": len(split_dfs["2020_ood"]),
            },
            "cold_start_stations_excluded": sherman_stations,
        },
        "dwell_assumption_test": dwell_results,
        "distribution_shift": compute_distribution_shift(
            split_dfs["2018_validation"], split_dfs["2019_test"], split_dfs["2020_ood"]
        ),
        "evaluations": {},
        "decision_metrics": {},
        "error_propagation": {},
        "slice_analysis": {},
    }

    # 5. Evaluate methods across horizons and splits
    methods = ["A0", "A1", "A2", "A3", "B1", "B2"]

    for h in horizons:
        results["evaluations"][h] = {}
        results["decision_metrics"][h] = {}
        results["error_propagation"][h] = {}

        # First run predictions for all splits
        for split_key, s_df in split_dfs.items():
            for m in methods:
                exp_occ, est_free, p_full = predict_availability_vectorized(
                    s_df, horizon=h, method=m, cap_col="capacity_time_varying",
                    empirical_rates=b1_rates, empirical_survival=emp_survival
                )
                s_df[f"{h}_{m}_exp_occ"] = exp_occ
                s_df[f"{h}_{m}_est_free"] = est_free
                s_df[f"{h}_{m}_prob_full"] = p_full

        # Fit Recalibrators strictly on 2018 Validation A0 and A3
        val_df = split_dfs["2018_validation"]
        y_val = val_df["true_full_event"].values

        # Isotonic on A0
        iso_a0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_a0.fit(val_df[f"{h}_A0_prob_full"].values, y_val)

        # Platt scaling on A0 (LogisticRegression)
        platt_a0 = LogisticRegression(C=1.0)
        logit_val_a0 = np.log(np.clip(val_df[f"{h}_A0_prob_full"].values, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(val_df[f"{h}_A0_prob_full"].values, 1e-5, 1.0 - 1e-5)))
        platt_a0.fit(logit_val_a0.reshape(-1, 1), y_val)

        # Apply recalibrators to all splits
        for split_key, s_df in split_dfs.items():
            cap_arr = s_df["capacity_time_varying"].values.astype(float)
            # C_isotonic
            p_iso = np.clip(iso_a0.predict(s_df[f"{h}_A0_prob_full"].values), 0.0, 1.0)
            s_df[f"{h}_C_isotonic_prob_full"] = p_iso
            s_df[f"{h}_C_isotonic_exp_occ"] = p_iso * cap_arr
            s_df[f"{h}_C_isotonic_est_free"] = np.maximum(0.0, cap_arr - s_df[f"{h}_C_isotonic_exp_occ"])

            # C_platt
            logit_s = np.log(np.clip(s_df[f"{h}_A0_prob_full"].values, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(s_df[f"{h}_A0_prob_full"].values, 1e-5, 1.0 - 1e-5)))
            p_platt = np.clip(platt_a0.predict_proba(logit_s.reshape(-1, 1))[:, 1], 0.0, 1.0)
            s_df[f"{h}_C_platt_prob_full"] = p_platt
            s_df[f"{h}_C_platt_exp_occ"] = p_platt * cap_arr
            s_df[f"{h}_C_platt_est_free"] = np.maximum(0.0, cap_arr - s_df[f"{h}_C_platt_exp_occ"])

            # Evaluate each method
            results["evaluations"][h][split_key] = {}
            results["decision_metrics"][h][split_key] = {}

            all_eval_methods = methods + ["C_isotonic", "C_platt"]
            for m in all_eval_methods:
                prob_col = f"{h}_{m}_prob_full"
                y_true = s_df["true_full_event"].values
                y_prob = s_df[prob_col].values
                y_true_free = s_df["true_free_points"].values
                y_est_free = s_df[f"{h}_{m}_est_free"].values if f"{h}_{m}_est_free" in s_df.columns else None

                cal_metrics = compute_calibration_metrics(y_true, y_prob)
                dec_metrics = compute_decision_metrics(
                    y_true,
                    y_prob,
                    y_true_free=y_true_free,
                    y_est_free=y_est_free,
                )

                # Bootstrap CIs for main test set
                if split_key == "2019_test" and m in ("A0", "B1", "C_isotonic"):
                    cis = block_bootstrap_confidence_intervals(s_df, "true_full_event", prob_col, n_boot=500, seed=seed)
                    cal_metrics["bootstrap_95_ci"] = cis

                results["evaluations"][h][split_key][m] = {
                    "calibration": cal_metrics,
                }
                results["decision_metrics"][h][split_key][m] = dec_metrics

            # Error propagation
            results["error_propagation"][h][split_key] = compute_error_propagation(s_df, horizon=h)

    # 6. Slice analysis: Single-port vs Multi-port, Peak vs Off-peak, Time-Varying vs Constant
    print("\n5. Running slice sensitivity analysis (capacity assumptions, port types, peak hours)...")
    test_19 = split_dfs["2019_test"]
    slice_data = []

    # Slice 1: Capacity assumption (time-varying vs constant)
    _, _, p_const = predict_availability_vectorized(test_19, horizon="1h", method="A0", cap_col="capacity_constant")
    cal_const = compute_calibration_metrics(test_19["true_full_event"].values, p_const)
    cal_tv = results["evaluations"]["1h"]["2019_test"]["A0"]["calibration"]

    slice_data.append({
        "slice_type": "Capacity Assumption",
        "slice_value": "Time-Varying (Actual)",
        "ece": cal_tv["ece"],
        "brier": cal_tv["brier_score"],
        "log_loss": cal_tv["log_loss"],
    })
    slice_data.append({
        "slice_type": "Capacity Assumption",
        "slice_value": "Constant (End-of-Data Assumption)",
        "ece": cal_const["ece"],
        "brier": cal_const["brier_score"],
        "log_loss": cal_const["log_loss"],
    })

    # Slice 2: Port Count (Single-port vs Multi-port)
    single_mask = test_19["capacity_time_varying"] == 1
    multi_mask = test_19["capacity_time_varying"] > 1

    p_a0_19 = test_19["1h_A0_prob_full"].values
    y_19 = test_19["true_full_event"].values

    cal_single = compute_calibration_metrics(y_19[single_mask], p_a0_19[single_mask])
    cal_multi = compute_calibration_metrics(y_19[multi_mask], p_a0_19[multi_mask])

    slice_data.append({
        "slice_type": "Port Architecture",
        "slice_value": "Single-Port Stations",
        "ece": cal_single["ece"],
        "brier": cal_single["brier_score"],
        "log_loss": cal_single["log_loss"],
    })
    slice_data.append({
        "slice_type": "Port Architecture",
        "slice_value": "Multi-Port Stations (2-3 ports)",
        "ece": cal_multi["ece"],
        "brier": cal_multi["brier_score"],
        "log_loss": cal_multi["log_loss"],
    })

    # Slice 3: Peak vs Off-Peak
    peak_mask = (test_19["hour"] >= 8) & (test_19["hour"] <= 18) & (~test_19["is_weekend"].astype(bool))
    offpeak_mask = ~peak_mask

    cal_peak = compute_calibration_metrics(y_19[peak_mask], p_a0_19[peak_mask])
    cal_offpeak = compute_calibration_metrics(y_19[offpeak_mask], p_a0_19[offpeak_mask])

    slice_data.append({
        "slice_type": "Temporal Regimes",
        "slice_value": "Weekday Peak (08:00-18:00)",
        "ece": cal_peak["ece"],
        "brier": cal_peak["brier_score"],
        "log_loss": cal_peak["log_loss"],
    })
    slice_data.append({
        "slice_type": "Temporal Regimes",
        "slice_value": "Off-Peak & Weekends",
        "ece": cal_offpeak["ece"],
        "brier": cal_offpeak["brier_score"],
        "log_loss": cal_offpeak["log_loss"],
    })

    results["slice_analysis"]["1h_2019_test"] = slice_data

    # 7. Save outputs
    print("\n6. Exporting tabular results and generating publication figures...")
    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved results JSON: {results_json_path}")

    # Export CSV tables
    # Table 1: Metrics by model and year (1h)
    rows_m = []
    for yr in ["2018_validation", "2019_test", "2020_ood"]:
        for m in all_eval_methods:
            c_meta = results["evaluations"]["1h"][yr][m]["calibration"]
            rows_m.append({
                "year_split": yr,
                "method": m,
                "ece": c_meta.get("ece"),
                "brier_score": c_meta.get("brier_score"),
                "brier_reliability": c_meta.get("brier_reliability"),
                "brier_resolution": c_meta.get("brier_resolution"),
                "log_loss": c_meta.get("log_loss"),
                "auroc": c_meta.get("auroc"),
                "auprc": c_meta.get("auprc"),
            })
    pd.DataFrame(rows_m).to_csv(os.path.join(output_dir, "metrics_by_model_and_year.csv"), index=False)

    # Table 2: Decision metrics by threshold (2019 test, 1h, A0 vs C_isotonic)
    rows_d = []
    for m in ["A0", "C_isotonic"]:
        m_data = results["decision_metrics"]["1h"]["2019_test"][m]
        sweep = m_data.get("threshold_sweep", m_data) if isinstance(m_data, dict) else m_data
        for row in sweep:
            r_copy = dict(row)
            r_copy["method"] = m
            rows_d.append(r_copy)
    pd.DataFrame(rows_d).to_csv(os.path.join(output_dir, "decision_metrics_by_threshold.csv"), index=False)

    # Table 2b: 3-Class Availability Confusion Matrix & Free Ports Error (2019 test, 1h)
    rows_cm = []
    for m in ["A0", "C_isotonic", "B1"]:
        m_dec = results["decision_metrics"]["1h"]["2019_test"].get(m, {})
        cm_data = m_dec.get("three_class_confusion_matrix")
        if cm_data:
            mat = cm_data["matrix"]
            rows_cm.append({
                "method": m,
                "overall_accuracy": cm_data["overall_accuracy"],
                "free_ports_mae": m_dec.get("free_ports_mae"),
                "free_ports_rmse": m_dec.get("free_ports_rmse"),
                "true_0_pred_0": mat[0][0], "true_0_pred_1": mat[0][1], "true_0_pred_2": mat[0][2],
                "true_1_pred_0": mat[1][0], "true_1_pred_1": mat[1][1], "true_1_pred_2": mat[1][2],
                "true_2_pred_0": mat[2][0], "true_2_pred_1": mat[2][1], "true_2_pred_2": mat[2][2],
                "recall_0_free": cm_data["per_class_recall"].get("0_free"),
                "recall_1_free": cm_data["per_class_recall"].get("1_free"),
                "recall_2_plus_free": cm_data["per_class_recall"].get("2_plus_free"),
            })
    if rows_cm:
        pd.DataFrame(rows_cm).to_csv(os.path.join(output_dir, "three_class_availability_confusion_matrix.csv"), index=False)

    # Table 2c: Error Binned by Arrival Forecast Error (2019 test, 1h)
    binned_errs = results["error_propagation"]["1h"]["2019_test"].get("error_binned_by_arrival_error", [])
    if binned_errs:
        pd.DataFrame(binned_errs).to_csv(os.path.join(output_dir, "error_binned_by_arrival_error.csv"), index=False)

    # Table 3: Error propagation
    rows_ep = []
    for yr in ["2018_validation", "2019_test", "2020_ood"]:
        ep = results["error_propagation"]["1h"][yr]
        rows_ep.append({
            "year_split": yr,
            "total_mae": ep["total_mae_occupied"],
            "arr_error_mae": ep["component_mae"]["arrival_forecast_error"],
            "dwell_error_mae": ep["component_mae"]["dwell_queueing_assumption_error"],
            "residual_error_mae": ep["component_mae"]["residual_structural_error"],
            "arr_error_share_pct": ep["variance_contributions_pct"]["arrival_forecast_error_share"],
            "dwell_error_share_pct": ep["variance_contributions_pct"]["dwell_queueing_assumption_error_share"],
            "residual_error_share_pct": ep["variance_contributions_pct"]["residual_structural_error_share"],
        })
    pd.DataFrame(rows_ep).to_csv(os.path.join(output_dir, "error_propagation_decomposition.csv"), index=False)

    # Table 4: Slice analysis
    pd.DataFrame(slice_data).to_csv(os.path.join(output_dir, "slice_analysis.csv"), index=False)

    # Table 5: Dwell KS test
    pd.DataFrame([
        {"split": k, **v} for k, v in dwell_results.items()
    ]).to_csv(os.path.join(output_dir, "dwell_ks_test_by_year.csv"), index=False)

    # Generate Figures
    generate_publication_figures(results, output_dir=output_dir)
    print("   Generated publication figures in reports/availability_calibration/")

    print("\n" + "=" * 80)
    print("CALIBRATION & ERROR PROPAGATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return results


# ==============================================================================
# 11. Parity Verification Test Function
# ==============================================================================

def verify_erlang_parity_with_service(n_samples: int = 10) -> bool:
    """
    Tests parity between vectorized Erlang-B calculations and availability_service.
    """
    test_cases = [
        {"servers": 1, "load": 0.2},
        {"servers": 1, "load": 1.5},
        {"servers": 2, "load": 0.4},
        {"servers": 2, "load": 1.8},
        {"servers": 3, "load": 2.5},
    ]

    for tc in test_cases:
        expected = compute_erlang_b(tc["servers"], tc["load"])
        vec = vectorized_erlang_b(np.array([tc["servers"]]), np.array([tc["load"]]))[0]
        assert abs(expected - vec) < 1e-6, f"Parity mismatch: {expected} vs {vec} for {tc}"
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Availability Calibration & Error Propagation Analysis")
    parser.add_argument("--horizon", choices=["1h", "24h", "both"], default="both", help="Evaluation horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping and sampling")
    args = parser.parse_args()

    h_list = ["1h", "24h"] if args.horizon == "both" else [args.horizon]
    run_calibration_experiment(horizons=h_list, seed=args.seed)
