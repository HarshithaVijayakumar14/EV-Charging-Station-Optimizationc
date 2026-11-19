"""
ml/train_probabilistic.py
Training pipeline for probabilistic EV arrival count models.
Strict chronological data split:
- Train: 2011-2017 (707k station-hours)
- Validation: 2018 (277k station-hours; model selection on Log Score / RPS)
- Retrain: 2011-2018 (985k station-hours; final production models)
- Test: 2019 (evaluated in ml/evaluate_probabilistic.py)
- OOD: 2020 (evaluated in ml/evaluate_probabilistic.py)

Models trained:
1. Poisson GLM (baseline)
2. Negative Binomial NB2 (dispersion modeling)
3. Hurdle Negative Binomial (sparse zero-heavy count modeling)
4. Empirical frequency baseline
Saves winning artifacts and metadata to models/.
"""

import json
import os
import time
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from ml.common import (
    FEATURES_1H,
    FEATURES_24H,
    FEATURES_CSV,
    MODELS_DIR,
    TARGET_COL,
    TabularPreprocessor,
    add_24h_ahead_features,
    load_hourly_features,
    sha256_file,
)
from ml.probabilistic_arrivals import (
    EmpiricalFrequencyBaseline,
    HurdleNegativeBinomialModel,
    NegativeBinomialModel,
    PoissonGLMModel,
    compute_pmf_metrics,
)


def compute_proper_scores(pmf: np.ndarray, y_true: np.ndarray, max_k: int = 12) -> Dict[str, float]:
    """
    Computes strictly proper scoring rules on validation set:
    - Mean Negative Log Score (-log P(Y = y_true))
    - Ranked Probability Score (RPS)
    - Zero Brier score
    - 90% central prediction interval coverage
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    n = len(y_true)
    k_clip = np.clip(y_true, 0, max_k)

    # 1. Log Score (negative log likelihood)
    probs_true = np.take_along_axis(pmf, k_clip[:, None], axis=1).ravel()
    probs_true = np.clip(probs_true, 1e-12, 1.0)
    neg_log_score = float(-np.mean(np.log(probs_true)))

    # 2. Ranked Probability Score (RPS)
    # RPS = sum_{k=0}^K (F(k) - 1(y <= k))^2
    cdf = np.cumsum(pmf, axis=1)
    indicators = (k_clip[:, None] <= np.arange(max_k + 1)[None, :]).astype(np.float64)
    rps = float(np.mean(np.sum((cdf - indicators) ** 2, axis=1)))

    # 3. Brier score for zero arrivals
    p_zero = pmf[:, 0]
    is_zero = (y_true == 0).astype(np.float64)
    zero_brier = float(np.mean((p_zero - is_zero) ** 2))

    # 4. Central 90% interval coverage
    q05 = np.argmax(cdf >= 0.05, axis=1)
    q95 = np.argmax(cdf >= 0.95, axis=1)
    in_interval = (k_clip >= q05) & (k_clip <= q95)
    coverage_90 = float(np.mean(in_interval))
    mean_width_90 = float(np.mean(q95 - q05))

    return {
        "log_score": round(neg_log_score, 4),
        "rps": round(rps, 4),
        "zero_brier": round(zero_brier, 4),
        "coverage_90": round(coverage_90, 4),
        "mean_width_90": round(mean_width_90, 4),
    }


def train_horizon_probabilistic(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    horizon_tag: str = "1h",
) -> Dict:
    print(f"\n{'='*25} PROBABILISTIC TRAINING: HORIZON {horizon_tag.upper()} {'='*25}", flush=True)
    t0 = time.time()

    train_df = df_all[df_all["year"] <= 2017].reset_index(drop=True)
    val_df = df_all[df_all["year"] == 2018].reset_index(drop=True)
    train_val_df = df_all[df_all["year"] <= 2018].reset_index(drop=True)

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_train_val = train_val_df[TARGET_COL].to_numpy(dtype=np.float64)

    print(f"[{horizon_tag}] Preprocessing features (strictly on training split)...", flush=True)
    preprocessor = TabularPreprocessor(feature_cols).fit(train_df)
    X_train = preprocessor.transform(train_df, impute=True)
    X_val = preprocessor.transform(val_df, impute=True)

    val_metrics = {}

    # 1. Baseline: Empirical Frequency
    print(f"[{horizon_tag}] 1. Fitting Empirical Frequency Baseline...", flush=True)
    emp_model = EmpiricalFrequencyBaseline(max_k=12).fit(train_df)
    emp_val_pmf = emp_model.predict_pmf(val_df)
    val_metrics["empirical_frequency"] = compute_proper_scores(emp_val_pmf, y_val)
    print(f"   Empirical Baseline Val 2018: LogScore={val_metrics['empirical_frequency']['log_score']}, RPS={val_metrics['empirical_frequency']['rps']}, 90%Cov={val_metrics['empirical_frequency']['coverage_90']}", flush=True)

    # 2. Baseline: Poisson GLM
    print(f"[{horizon_tag}] 2. Fitting Poisson GLM Model...", flush=True)
    poisson_model = PoissonGLMModel(alpha=1e-4, max_iter=300).fit(X_train, y_train)
    poisson_val_pmf = poisson_model.predict_pmf(X_val)
    val_metrics["poisson_glm"] = compute_proper_scores(poisson_val_pmf, y_val)
    print(f"   Poisson GLM Val 2018: LogScore={val_metrics['poisson_glm']['log_score']}, RPS={val_metrics['poisson_glm']['rps']}, 90%Cov={val_metrics['poisson_glm']['coverage_90']}", flush=True)

    # 3. Main: Negative Binomial (NB2)
    print(f"[{horizon_tag}] 3. Fitting Negative Binomial Model...", flush=True)
    nb_model = NegativeBinomialModel(alpha_reg=1e-4, max_iter=300).fit(X_train, y_train)
    nb_val_pmf = nb_model.predict_pmf(X_val)
    val_metrics["negative_binomial"] = compute_proper_scores(nb_val_pmf, y_val)
    print(f"   Negative Binomial Val 2018: LogScore={val_metrics['negative_binomial']['log_score']}, RPS={val_metrics['negative_binomial']['rps']}, alpha={nb_model.dispersion_alpha:.4f}", flush=True)

    # 4. Hurdle / Zero-Inflated Negative Binomial
    print(f"[{horizon_tag}] 4. Fitting Hurdle Negative Binomial Model...", flush=True)
    hurdle_model = HurdleNegativeBinomialModel(C_zero=1.0, alpha_reg=1e-4, max_iter=300).fit(X_train, y_train)
    hurdle_val_pmf = hurdle_model.predict_pmf(X_val)
    val_metrics["hurdle_nb"] = compute_proper_scores(hurdle_val_pmf, y_val)
    print(f"   Hurdle NB Val 2018: LogScore={val_metrics['hurdle_nb']['log_score']}, RPS={val_metrics['hurdle_nb']['rps']}, ZeroBrier={val_metrics['hurdle_nb']['zero_brier']}", flush=True)

    # Winner selection based on lowest validation Log Score (strictly proper rule)
    candidate_keys = ["poisson_glm", "negative_binomial", "hurdle_nb"]
    winner_key = min(candidate_keys, key=lambda k: val_metrics[k]["log_score"])
    print(f"[{horizon_tag}] WINNER on Validation 2018: '{winner_key}' (Log Score: {val_metrics[winner_key]['log_score']})", flush=True)

    # Final Retrain on 2011-2018 (Train + Val)
    print(f"[{horizon_tag}] Final retrain of '{winner_key}' on 2011-2018 (985,204 rows)...", flush=True)
    final_prep = TabularPreprocessor(feature_cols).fit(train_val_df)
    X_train_val = final_prep.transform(train_val_df, impute=True)

    if winner_key == "hurdle_nb":
        final_model = HurdleNegativeBinomialModel(C_zero=1.0, alpha_reg=1e-4, max_iter=300).fit(X_train_val, y_train_val)
    elif winner_key == "negative_binomial":
        final_model = NegativeBinomialModel(alpha_reg=1e-4, max_iter=300).fit(X_train_val, y_train_val)
    else:
        final_model = PoissonGLMModel(alpha=1e-4, max_iter=300).fit(X_train_val, y_train_val)

    # Save candidate models and winning model bundle
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_bundle = {
        "model": final_model,
        "preprocessor": final_prep,
        "model_type": winner_key,
        "features": feature_cols,
        "horizon": horizon_tag,
        "validation_metrics": val_metrics,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": int(len(train_val_df)),
    }
    model_save_path = os.path.join(MODELS_DIR, f"best_probabilistic_model_{horizon_tag}.joblib")
    joblib.dump(best_model_bundle, model_save_path, compress=3)
    print(f"[{horizon_tag}] Saved best probabilistic model bundle to {model_save_path} ({os.path.getsize(model_save_path):,} bytes)", flush=True)

    # Also save the candidate models for 1h horizon
    if horizon_tag == "1h":
        joblib.dump({"model": poisson_model, "preprocessor": preprocessor}, os.path.join(MODELS_DIR, "poisson_glm_1h.joblib"), compress=3)
        joblib.dump({"model": nb_model, "preprocessor": preprocessor}, os.path.join(MODELS_DIR, "negative_binomial_1h.joblib"), compress=3)
        joblib.dump({"model": hurdle_model, "preprocessor": preprocessor}, os.path.join(MODELS_DIR, "hurdle_nb_1h.joblib"), compress=3)

    return {
        "horizon": horizon_tag,
        "winner": winner_key,
        "validation_2018_metrics": val_metrics,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main():
    print("=" * 70, flush=True)
    print("STARTING PROBABILISTIC ARRIVAL MODEL TRAINING", flush=True)
    print("=" * 70, flush=True)
    total_start = time.time()

    print(f"Loading hourly feature dataset from: {FEATURES_CSV} ...", flush=True)
    df_all = load_hourly_features(FEATURES_CSV)
    print(f"Loaded {len(df_all):,} station-hours. Features shape: {df_all.shape}", flush=True)

    # 1-Hour-Ahead Training
    res_1h = train_horizon_probabilistic(df_all, FEATURES_1H, horizon_tag="1h")

    # 24-Hour-Ahead Training
    print("\nPreparing strictly causal 24h-ahead features...", flush=True)
    df_24h = add_24h_ahead_features(df_all)
    res_24h = train_horizon_probabilistic(df_24h, FEATURES_24H, horizon_tag="24h")

    metadata = {
        "task": "Probabilistic EV Arrival Count Forecasting",
        "dataset_hash": sha256_file(FEATURES_CSV),
        "primary_1h_ahead": res_1h,
        "secondary_24h_ahead": res_24h,
        "total_training_time_sec": round(time.time() - total_start, 2),
    }
    meta_path = os.path.join(MODELS_DIR, "probabilistic_training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}", flush=True)
    print("PROBABILISTIC ARRIVAL MODEL TRAINING COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
