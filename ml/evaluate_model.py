import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from ml.common import (
    CLEANED_CSV,
    EVAL_DIR,
    FEATURES_1H,
    FEATURES_24H,
    FEATURES_CSV,
    MODELS_DIR,
    RAW_CSV,
    TARGET_COL,
    add_24h_ahead_features,
    compute_metrics,
    load_hourly_features,
    sha256_file,
)
from ml.predict_demand import predict_from_dataframe

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.02)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#CCCCCC",
    "axes.titleweight": "bold",
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "grid.alpha": 0.45,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

MODEL_ORDER = ["historical_mean", "seasonal_naive", "random_forest", "xgboost"]
MODEL_LABELS = {
    "historical_mean": "Historical Mean",
    "seasonal_naive": "Seasonal Naive",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "historical_mean": "#8D99AE",
    "seasonal_naive": "#E9C46A",
    "random_forest": "#2A9D8F",
    "xgboost": "#1F6F8B",
}


def compute_detailed_slices(df_eval: pd.DataFrame, y_pred: np.ndarray, seen_stations: set) -> dict:
    y_true = df_eval[TARGET_COL].to_numpy(dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)

    out = {"overall": compute_metrics(y_true, y_p)}

    # Zero vs Non-zero arrival hours
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    out["zero_arrival_hours"] = compute_metrics(y_true[zero_mask], y_p[zero_mask])
    out["nonzero_arrival_hours"] = compute_metrics(y_true[nonzero_mask], y_p[nonzero_mask])

    # Seen (historical) vs New stations
    seen_mask = df_eval["station_id"].isin(seen_stations).to_numpy()
    new_mask = ~seen_mask
    out["historical_stations"] = compute_metrics(y_true[seen_mask], y_p[seen_mask]) if seen_mask.any() else None
    out["new_stations"] = compute_metrics(y_true[new_mask], y_p[new_mask]) if new_mask.any() else None

    # Weekday vs Weekend
    wkday_mask = df_eval["is_weekend"].to_numpy() == 0
    wkend_mask = ~wkday_mask
    out["weekday"] = compute_metrics(y_true[wkday_mask], y_p[wkday_mask])
    out["weekend"] = compute_metrics(y_true[wkend_mask], y_p[wkend_mask])

    # By hour of day (0-23)
    by_hour = {}
    hrs = df_eval["hour"].to_numpy()
    for h in range(24):
        m = hrs == h
        if m.any():
            met = compute_metrics(y_true[m], y_p[m])
            met["actual_mean"] = round(float(y_true[m].mean()), 4)
            met["pred_mean"] = round(float(y_p[m].mean()), 4)
            by_hour[str(h)] = met
    out["by_hour"] = by_hour

    # By station
    by_station = []
    tmp = pd.DataFrame({
        "station_id": df_eval["station_id"].values,
        "y_true": y_true,
        "y_pred": y_p,
    })
    for st, g in tmp.groupby("station_id"):
        met = compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        met["station_id"] = st
        met["is_historical"] = st in seen_stations
        by_station.append(met)
    by_station.sort(key=lambda x: x["actual_sum"], reverse=True)
    out["by_station"] = by_station

    return out


def compute_feature_importance_bundle(bundle: dict, df_sample: pd.DataFrame) -> dict:
    model = bundle["model"]
    prep = bundle["preprocessor"]
    impute = bool(bundle.get("impute_missing", True))
    feat_cols = bundle["feature_cols"]

    X_s = prep.transform(df_sample, impute=impute)
    y_s = df_sample[TARGET_COL].to_numpy(dtype=np.float64)

    # Native model importance
    if bundle["model_name"] == "xgboost":
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")
        native_vals = [float(gain_dict.get(f"f{i}", 0.0)) for i in range(len(feat_cols))]
    elif hasattr(model, "feature_importances_"):
        native_vals = [float(v) for v in model.feature_importances_]
    else:
        native_vals = [0.0] * len(feat_cols)

    native_sum = sum(native_vals) or 1.0
    native_pct = [round(v / native_sum * 100.0, 3) for v in native_vals]

    # Permutation importance (increase in RMSE when feature is shuffled; neg_root_mean_squared_error)
    perm = permutation_importance(
        model,
        X_s,
        y_s,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    rows = []
    for i, col in enumerate(feat_cols):
        rows.append({
            "feature": col,
            "perm_rmse_increase_mean": round(float(perm.importances_mean[i]), 6),
            "perm_rmse_increase_std": round(float(perm.importances_std[i]), 6),
            "native_importance_pct": native_pct[i],
        })
    rows.sort(key=lambda x: x["perm_rmse_increase_mean"], reverse=True)
    return {
        "model_name": bundle["model_name"],
        "horizon": bundle["horizon"],
        "sample_rows": int(len(df_sample)),
        "features_ranked_by_permutation_rmse": rows,
    }


def make_charts(
    eval_results: dict,
    meta: dict,
    df_2019: pd.DataFrame,
    preds_2019_1h: dict,
    preds_2019_24h: dict,
    df_2020: pd.DataFrame,
    preds_2020_1h: dict,
    winner_1h: str,
    winner_24h: str,
    fi_1h: dict,
    fi_24h: dict,
):
    os.makedirs(EVAL_DIR, exist_ok=True)
    short_st = lambda s: s.replace("PALO ALTO CA / ", "")

    # =========================================================================
    # CHART 1: Model Comparison Across Val 2018, Test 2019, OOD 2020 (1h & 24h)
    # =========================================================================
    print("Generating 01_model_comparison_metrics.png...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    splits = ["Val 2018\n(1h)", "Test 2019\n(1h)", "OOD 2020\n(1h)", "Test 2019\n(24h)"]
    x = np.arange(len(splits))
    w = 0.19

    for ax_idx, (metric_key, metric_title, fmt_str) in enumerate([
        ("mae", "MAE (Lower is Better)", "{:.3f}"),
        ("rmse", "RMSE (Lower is Better)", "{:.3f}"),
        ("r2", "R² (Higher is Better)", "{:.3f}"),
    ]):
        ax = axes[ax_idx]
        for m_i, m_key in enumerate(MODEL_ORDER):
            vals = [
                meta["primary_1h_ahead"]["validation_2018_results"][m_key]["overall"][metric_key],
                eval_results["primary_1h_ahead"]["test_2019"][m_key]["overall"][metric_key],
                eval_results["primary_1h_ahead"]["ood_2020"][m_key]["overall"][metric_key],
                eval_results["secondary_24h_ahead"]["test_2019"][m_key]["overall"][metric_key],
            ]
            offset = (m_i - 1.5) * w
            bars = ax.bar(x + offset, vals, width=w, color=MODEL_COLORS[m_key], label=MODEL_LABELS[m_key], edgecolor="white")
            for b, v in zip(bars, vals):
                y_pos = v + (0.006 if v >= 0 else -0.025)
                va = "bottom" if v >= 0 else "top"
                ax.text(b.get_x() + b.get_width() / 2, y_pos, fmt_str.format(v), ha="center", va=va, fontsize=7.2, rotation=90)
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=9.5)
        if metric_key == "r2":
            ax.axhline(0, color="#444444", linewidth=0.9)
            ymin = min(-0.08, ax.get_ylim()[0] * 1.25)
            ax.set_ylim(ymin, max(0.55, ax.get_ylim()[1] * 1.28))
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        if ax_idx == 0:
            ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.5)

    fig.suptitle("Model Comparison Across Chronological Splits (1-Hour-Ahead & 24-Hour-Ahead)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "01_model_comparison_metrics.png"))
    plt.close(fig)

    # =========================================================================
    # CHART 2: Actual vs Predicted in 2019 (Daily Full Year + Hourly Sample Week)
    # =========================================================================
    print("Generating 02_actual_vs_predicted_2019.png...", flush=True)
    d19 = pd.DataFrame({
        "pt_dt": pd.to_datetime(df_2019["hour_timestamp_pt"].str.slice(0, 19)),
        "actual": df_2019[TARGET_COL].to_numpy(),
        "hist_mean": preds_2019_1h["historical_mean"],
        "winner_1h": preds_2019_1h[winner_1h],
        "winner_24h": preds_2019_24h[winner_24h],
    })
    d19["date"] = d19["pt_dt"].dt.date
    daily19 = d19.groupby("date")[["actual", "hist_mean", "winner_1h", "winner_24h"]].sum()
    daily19_roll = daily19.rolling(7, min_periods=1).mean()

    # Sample busy week: Oct 7 - Oct 13, 2019 (Mon-Sun)
    wk_mask = (d19["pt_dt"] >= "2019-10-07 00:00:00") & (d19["pt_dt"] < "2019-10-14 00:00:00")
    wk_hr = d19[wk_mask].groupby("pt_dt")[["actual", "winner_1h", "winner_24h", "hist_mean"]].sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.6))
    ax1.plot(daily19_roll.index, daily19_roll["actual"], color="#111111", lw=2.2, label="Actual (7-Day Moving Avg)")
    ax1.plot(daily19_roll.index, daily19_roll["winner_1h"], color=MODEL_COLORS[winner_1h], lw=1.9, label=f"{MODEL_LABELS[winner_1h]} (1h-Ahead)")
    ax1.plot(daily19_roll.index, daily19_roll["winner_24h"], color="#E07A5F", lw=1.6, linestyle="--", label=f"{MODEL_LABELS[winner_24h]} (24h-Ahead)")
    ax1.plot(daily19_roll.index, daily19_roll["hist_mean"], color="#8D99AE", lw=1.4, linestyle=":", label="Historical Mean Baseline")
    ax1.set_title("A. Main Test 2019 — Network-Wide Daily Sessions (7-Day Moving Average)")
    ax1.set_ylabel("Sessions per Day (All 33 Stations)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    ax2.plot(wk_hr.index, wk_hr["actual"], color="#111111", lw=2.0, label="Actual Hourly Arrivals")
    ax2.plot(wk_hr.index, wk_hr["winner_1h"], color=MODEL_COLORS[winner_1h], lw=1.8, alpha=0.92, label=f"{MODEL_LABELS[winner_1h]} (1h-Ahead)")
    ax2.plot(wk_hr.index, wk_hr["winner_24h"], color="#E07A5F", lw=1.5, linestyle="--", alpha=0.9, label=f"{MODEL_LABELS[winner_24h]} (24h-Ahead)")
    ax2.set_title("B. Sample Week Hourly Profile (Oct 7–13, 2019, Mon–Sun — Network-Wide)")
    ax2.set_ylabel("Sessions per Hour")
    ax2.set_xlabel("Date & Hour (Pacific Local Time)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    fig.suptitle("Actual vs. Predicted Charging Demand — Main Test Set (2019)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "02_actual_vs_predicted_2019.png"))
    plt.close(fig)

    # =========================================================================
    # CHART 3: 2020 OOD Stress Test (COVID-19 Shock Adaptation)
    # =========================================================================
    print("Generating 03_actual_vs_predicted_2020_ood.png...", flush=True)
    d20 = pd.DataFrame({
        "date": pd.to_datetime(df_2020["hour_timestamp_pt"].str.slice(0, 10)),
        "actual": df_2020[TARGET_COL].to_numpy(),
        "hist_mean": preds_2020_1h["historical_mean"],
        "seas_naive": preds_2020_1h["seasonal_naive"],
        "rf": preds_2020_1h["random_forest"],
        "xgb": preds_2020_1h["xgboost"],
    })
    daily20 = d20.groupby("date")[["actual", "hist_mean", "seas_naive", "rf", "xgb"]].sum().rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.plot(daily20.index, daily20["actual"], color="#111111", lw=2.4, label="Actual 2020 Demand (7-Day Avg)")
    ax.plot(daily20.index, daily20["xgb"], color=MODEL_COLORS["xgboost"], lw=1.9, label="XGBoost (1h-Ahead)")
    ax.plot(daily20.index, daily20["rf"], color=MODEL_COLORS["random_forest"], lw=1.7, linestyle="--", label="Random Forest (1h-Ahead)")
    ax.plot(daily20.index, daily20["seas_naive"], color=MODEL_COLORS["seasonal_naive"], lw=1.5, linestyle="-.", label="Seasonal Naive (24h/168h)")
    ax.plot(daily20.index, daily20["hist_mean"], color=MODEL_COLORS["historical_mean"], lw=1.6, linestyle=":", label="Historical Mean Baseline (Static 2011–2018)")
    ax.axvline(pd.Timestamp("2020-03-16"), color="#D62828", linestyle="--", lw=1.3, alpha=0.8)
    ax.text(pd.Timestamp("2020-03-20"), daily20["hist_mean"].max() * 0.92, "Bay Area COVID-19\nShelter-in-Place\n(Mid-March 2020)", color="#D62828", fontsize=9, fontweight="bold")
    ax.set_title("2020 Out-of-Distribution Stress Test: Model Adaptation to COVID-19 Demand Collapse")
    ax.set_ylabel("Network-Wide Sessions per Day (7-Day Moving Avg)")
    ax.set_xlabel("Date (2020)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "03_actual_vs_predicted_2020_ood.png"))
    plt.close(fig)

    # =========================================================================
    # CHART 4: Error Breakdown Slices on 2019 Main Test (Hourly, Wkday/Wknd, Zero/Non-Zero)
    # =========================================================================
    print("Generating 04_error_breakdown_slices.png...", flush=True)
    w_res19 = eval_results["primary_1h_ahead"]["test_2019"][winner_1h]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # Panel A: Actual vs Predicted Mean by Hour of Day (2019) + Hourly MAE
    ax = axes[0]
    hrs = list(range(24))
    act_hr = [w_res19["by_hour"][str(h)]["actual_mean"] for h in hrs]
    pred_hr = [w_res19["by_hour"][str(h)]["pred_mean"] for h in hrs]
    mae_hr = [w_res19["by_hour"][str(h)]["mae"] for h in hrs]
    ax.plot(hrs, act_hr, color="#111111", marker="o", lw=2.0, markersize=4, label="Actual Mean")
    ax.plot(hrs, pred_hr, color=MODEL_COLORS[winner_1h], marker="s", lw=1.9, markersize=4, linestyle="--", label=f"Predicted Mean ({MODEL_LABELS[winner_1h]})")
    ax.bar(hrs, mae_hr, color="#E07A5F", alpha=0.28, label="Hourly MAE")
    ax.set_title("A. 2019 Hourly Calibration & MAE")
    ax.set_xlabel("Hour of Day (Pacific Time)")
    ax.set_ylabel("Sessions / Station-Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.2)

    # Panel B: Weekday vs Weekend RMSE across models
    ax = axes[1]
    cats = ["Weekday", "Weekend"]
    x = np.arange(2)
    w_b = 0.19
    for i, m_key in enumerate(MODEL_ORDER):
        r = eval_results["primary_1h_ahead"]["test_2019"][m_key]
        vals = [r["weekday"]["rmse"], r["weekend"]["rmse"]]
        bars = ax.bar(x + (i - 1.5) * w_b, vals, width=w_b, color=MODEL_COLORS[m_key], label=MODEL_LABELS[m_key], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_title("B. 2019 Weekday vs. Weekend RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8)

    # Panel C: Zero vs Non-Zero Arrival Hours MAE across models
    ax = axes[2]
    cats = ["Zero-Arrival\nHours (85.2%)", "Non-Zero\nHours (14.8%)"]
    for i, m_key in enumerate(MODEL_ORDER):
        r = eval_results["primary_1h_ahead"]["test_2019"][m_key]
        vals = [r["zero_arrival_hours"]["mae"], r["nonzero_arrival_hours"]["mae"]]
        bars = ax.bar(x + (i - 1.5) * w_b, vals, width=w_b, color=MODEL_COLORS[m_key], label=MODEL_LABELS[m_key], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_title("C. 2019 Zero vs. Non-Zero Arrival MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("MAE")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)

    fig.suptitle("Diagnostic Error Slices — Main Test Set (2019, 1-Hour-Ahead)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "04_error_breakdown_slices.png"))
    plt.close(fig)

    # =========================================================================
    # CHART 5: Station-Level Performance (2019 Top 20 + 2020 Seen vs New Stations)
    # =========================================================================
    print("Generating 05_station_level_performance.png...", flush=True)
    st19 = w_res19["by_station"][:20][::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.35, 1]})

    y_pos = np.arange(len(st19))
    h_bar = 0.38
    ax1.barh(y_pos + h_bar / 2, [r["actual_sum"] for r in st19], height=h_bar, color="#264653", label="Actual 2019 Sessions")
    ax1.barh(y_pos - h_bar / 2, [r["predicted_sum"] for r in st19], height=h_bar, color=MODEL_COLORS[winner_1h], alpha=0.85, label=f"Predicted ({MODEL_LABELS[winner_1h]})")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([short_st(r["station_id"]) for r in st19], fontsize=8.8)
    ax1.set_xlabel("Total Annual Sessions in 2019")
    ax1.set_title("A. Top 20 Stations: Actual vs. Predicted Annual Volume (2019)")
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)

    # Panel B: Historical vs New Stations RMSE on Val 2018 and OOD 2020
    ax = ax2
    groups = ["Val 2018\nHistorical (27)", "Val 2018\nNew (6)", "OOD 2020\nHistorical (33)", "OOD 2020\nNew Sherman (13)"]
    xg = np.arange(len(groups))
    w_g = 0.19
    for i, m_key in enumerate(MODEL_ORDER):
        v18 = meta["primary_1h_ahead"]["validation_2018_results"][m_key]
        o20 = eval_results["primary_1h_ahead"]["ood_2020"][m_key]
        vals = [
            v18["historical_stations"]["rmse"],
            v18["new_stations"]["rmse"],
            o20["historical_stations"]["rmse"],
            o20["new_stations"]["rmse"],
        ]
        bars = ax.bar(xg + (i - 1.5) * w_g, vals, width=w_g, color=MODEL_COLORS[m_key], label=MODEL_LABELS[m_key], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax.set_title("B. Historical vs. Newly Opened Stations RMSE")
    ax.set_xticks(xg)
    ax.set_xticklabels(groups, fontsize=8.8)
    ax.set_ylabel("RMSE (Lower is Better)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.2)

    fig.suptitle("Station-Level Accuracy & Cold-Start Generalization", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "05_station_level_performance.png"))
    plt.close(fig)

    # =========================================================================
    # CHART 6: Feature Importance (Permutation RMSE Increase — 1h vs 24h)
    # =========================================================================
    print("Generating 06_feature_importance.png...", flush=True)
    top15_1h = fi_1h["features_ranked_by_permutation_rmse"][:15][::-1]
    top15_24h = fi_24h["features_ranked_by_permutation_rmse"][:15][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8))
    ax1.barh(
        [r["feature"] for r in top15_1h],
        [r["perm_rmse_increase_mean"] for r in top15_1h],
        xerr=[r["perm_rmse_increase_std"] for r in top15_1h],
        color=MODEL_COLORS[winner_1h], edgecolor="white", capsize=2.5,
    )
    ax1.set_title(f"A. Top 15 Features — 1-Hour-Ahead ({MODEL_LABELS[winner_1h]})")
    ax1.set_xlabel("Permutation Importance (Mean RMSE Increase When Shuffled)")

    ax2.barh(
        [r["feature"] for r in top15_24h],
        [r["perm_rmse_increase_mean"] for r in top15_24h],
        xerr=[r["perm_rmse_increase_std"] for r in top15_24h],
        color="#E07A5F", edgecolor="white", capsize=2.5,
    )
    ax2.set_title(f"B. Top 15 Features — 24-Hour-Ahead ({MODEL_LABELS[winner_24h]})")
    ax2.set_xlabel("Permutation Importance (Mean RMSE Increase When Shuffled)")

    fig.suptitle("Predictive Feature Importance on 2019 Test Sample (Associational — Not Causal Proof)", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "06_feature_importance.png"))
    plt.close(fig)


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    print("Verifying upstream dataset hashes before evaluation...", flush=True)
    raw_sha = sha256_file(RAW_CSV)
    clean_sha = sha256_file(CLEANED_CSV)
    feat_sha = sha256_file(FEATURES_CSV)

    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    winner_1h = meta["primary_1h_ahead"]["selected_winner_on_val_2018"]
    winner_24h = meta["secondary_24h_ahead"]["selected_winner_on_val_2018"]
    print(f"Loaded training metadata. Val-selected winners -> 1h: {winner_1h}, 24h: {winner_24h}", flush=True)

    print("Loading hourly features for untouched 2019 Test and 2020 OOD evaluation...", flush=True)
    df = load_hourly_features(FEATURES_CSV)
    df_24 = add_24h_ahead_features(df)

    seen_11_18 = set(df[df["year"] <= 2018]["station_id"].unique())
    df_2019 = df[df["year"] == 2019].reset_index(drop=True)
    df_2020 = df[df["year"] == 2020].reset_index(drop=True)
    df_2019_24 = df_24[df_24["year"] == 2019].reset_index(drop=True)
    df_2020_24 = df_24[df_24["year"] == 2020].reset_index(drop=True)

    model_files = {
        "historical_mean": "baseline_historical_mean",
        "seasonal_naive": "baseline_seasonal_naive",
        "random_forest": "random_forest",
        "xgboost": "xgboost",
    }

    eval_results = {
        "primary_1h_ahead": {"winner_selected_on_val_2018": winner_1h, "test_2019": {}, "ood_2020": {}},
        "secondary_24h_ahead": {"winner_selected_on_val_2018": winner_24h, "test_2019": {}, "ood_2020": {}},
    }

    preds_2019_1h = {}
    preds_2020_1h = {}
    preds_2019_24h = {}
    preds_2020_24h = {}

    # Global constant mean reference (retrained on 2011-2018)
    gmean_11_18 = float(df[df["year"] <= 2018][TARGET_COL].mean())
    eval_results["global_constant_mean_2011_2018_ref"] = {
        "train_val_mean": round(gmean_11_18, 5),
        "test_2019": compute_metrics(df_2019[TARGET_COL].to_numpy(), np.full(len(df_2019), gmean_11_18)),
        "ood_2020": compute_metrics(df_2020[TARGET_COL].to_numpy(), np.full(len(df_2020), gmean_11_18)),
    }

    print("\n--- Evaluating Primary 1-Hour-Ahead Models on 2019 Test & 2020 OOD ---", flush=True)
    for m_key in MODEL_ORDER:
        m_path = os.path.join(MODELS_DIR, f"{model_files[m_key]}_1h.joblib")
        p19 = predict_from_dataframe(df_2019, horizon="1h", model_path=m_path)
        p20 = predict_from_dataframe(df_2020, horizon="1h", model_path=m_path)
        preds_2019_1h[m_key] = p19
        preds_2020_1h[m_key] = p20

        r19 = compute_detailed_slices(df_2019, p19, seen_11_18)
        r20 = compute_detailed_slices(df_2020, p20, seen_11_18)
        eval_results["primary_1h_ahead"]["test_2019"][m_key] = r19
        eval_results["primary_1h_ahead"]["ood_2020"][m_key] = r20
        print(
            f"  [1h] {m_key:16s} | 2019: MAE={r19['overall']['mae']:.5f} RMSE={r19['overall']['rmse']:.5f} R2={r19['overall']['r2']:.5f} "
            f"| 2020: MAE={r20['overall']['mae']:.5f} RMSE={r20['overall']['rmse']:.5f} R2={r20['overall']['r2']:.5f}",
            flush=True,
        )

    print("\n--- Evaluating Secondary 24-Hour-Ahead Models on 2019 Test & 2020 OOD ---", flush=True)
    for m_key in MODEL_ORDER:
        m_path = os.path.join(MODELS_DIR, f"{model_files[m_key]}_24h.joblib")
        p19 = predict_from_dataframe(df_2019_24, horizon="24h", model_path=m_path)
        p20 = predict_from_dataframe(df_2020_24, horizon="24h", model_path=m_path)
        preds_2019_24h[m_key] = p19
        preds_2020_24h[m_key] = p20

        r19 = compute_detailed_slices(df_2019_24, p19, seen_11_18)
        r20 = compute_detailed_slices(df_2020_24, p20, seen_11_18)
        eval_results["secondary_24h_ahead"]["test_2019"][m_key] = r19
        eval_results["secondary_24h_ahead"]["ood_2020"][m_key] = r20
        print(
            f"  [24h] {m_key:15s} | 2019: MAE={r19['overall']['mae']:.5f} RMSE={r19['overall']['rmse']:.5f} R2={r19['overall']['r2']:.5f} "
            f"| 2020: MAE={r20['overall']['mae']:.5f} RMSE={r20['overall']['rmse']:.5f} R2={r20['overall']['r2']:.5f}",
            flush=True,
        )

    # Feature importance on fixed-seed 60,000-row sample of 2019 Test for winning 1h & 24h models
    print("\nComputing Permutation & Native Feature Importance on 2019 Test sample (n=60,000)...", flush=True)
    rng = np.random.RandomState(42)
    sample_idx = np.sort(rng.choice(len(df_2019), size=min(60000, len(df_2019)), replace=False))

    best_1h_bundle = joblib.load(os.path.join(MODELS_DIR, "best_model_1h.joblib"))
    best_24h_bundle = joblib.load(os.path.join(MODELS_DIR, "best_model_24h.joblib"))
    fi_1h = compute_feature_importance_bundle(best_1h_bundle, df_2019.iloc[sample_idx])
    fi_24h = compute_feature_importance_bundle(best_24h_bundle, df_2019_24.iloc[sample_idx])
    eval_results["feature_importance_1h"] = fi_1h
    eval_results["feature_importance_24h"] = fi_24h

    # Generate all 6 comparison & diagnostic charts
    make_charts(
        eval_results=eval_results,
        meta=meta,
        df_2019=df_2019,
        preds_2019_1h=preds_2019_1h,
        preds_2019_24h=preds_2019_24h,
        df_2020=df_2020,
        preds_2020_1h=preds_2020_1h,
        winner_1h=winner_1h,
        winner_24h=winner_24h,
        fi_1h=fi_1h,
        fi_24h=fi_24h,
    )

    # Verify upstream files untouched
    assert sha256_file(RAW_CSV) == raw_sha, "Raw CSV changed!"
    assert sha256_file(CLEANED_CSV) == clean_sha, "Cleaned CSV changed!"
    assert sha256_file(FEATURES_CSV) == feat_sha, "Features CSV changed!"

    out_json = os.path.join(EVAL_DIR, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEVALUATION COMPLETE! Results saved to {out_json} and 6 charts saved to {EVAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
