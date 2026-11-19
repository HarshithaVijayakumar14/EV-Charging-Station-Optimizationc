import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml.common import (
    CLEANED_CSV,
    FEATURES_1H,
    FEATURES_24H,
    FEATURES_CSV,
    MODELS_DIR,
    RAW_CSV,
    TARGET_COL,
    HistoricalMeanBaseline,
    SeasonalNaiveBaseline,
    TabularPreprocessor,
    add_24h_ahead_features,
    compute_metrics,
    load_hourly_features,
    sha256_file,
    verify_24h_no_future_leakage,
)


def eval_val_with_station_slices(y_true, y_pred, stations, seen_stations_set):
    overall = compute_metrics(y_true, y_pred)
    seen_mask = stations.isin(seen_stations_set).to_numpy()
    new_mask = ~seen_mask
    res = {"overall": overall}
    if seen_mask.any():
        res["historical_stations"] = compute_metrics(y_true[seen_mask], y_pred[seen_mask])
    if new_mask.any():
        res["new_stations"] = compute_metrics(y_true[new_mask], y_pred[new_mask])
    return res


def run_horizon_training(
    df_all: pd.DataFrame,
    feature_cols: list,
    horizon_tag: str,
):
    print(f"\n================ TRAINING HORIZON: {horizon_tag.upper()} ================", flush=True)
    train_df = df_all[df_all["year"] <= 2017].reset_index(drop=True)
    val_df = df_all[df_all["year"] == 2018].reset_index(drop=True)
    train_val_df = df_all[df_all["year"] <= 2018].reset_index(drop=True)

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_train_val = train_val_df[TARGET_COL].to_numpy(dtype=np.float32)

    seen_11_17 = set(train_df["station_id"].unique())
    val_stations = val_df["station_id"]

    # Strict training-only preprocessor for 2011-2017 -> 2018 tuning
    prep_tune = TabularPreprocessor(feature_cols).fit(train_df)
    X_train_imp = prep_tune.transform(train_df, impute=True)
    X_val_imp = prep_tune.transform(val_df, impute=True)
    # Native NaN arrays for XGBoost (station_encoded still uses training-only mapping)
    X_train_xgb = prep_tune.transform(train_df, impute=False)
    X_val_xgb = prep_tune.transform(val_df, impute=False)

    tuning_log = {
        "horizon": horizon_tag,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "split_rows": {
            "train_2011_2017": int(len(train_df)),
            "val_2018": int(len(val_df)),
            "train_val_2011_2018": int(len(train_val_df)),
        },
        "validation_2018_results": {},
    }

    # -------------------------------------------------------------------------
    # 1. Historical Mean Baseline (plus global constant mean reference)
    # -------------------------------------------------------------------------
    print(f"[{horizon_tag}] Tuning/Evaluating Baseline 1: Historical Mean...", flush=True)
    global_mean_val = np.full(len(y_val), float(y_train.mean()), dtype=np.float64)
    tuning_log["validation_2018_results"]["global_constant_mean_ref"] = eval_val_with_station_slices(
        y_val, global_mean_val, val_stations, seen_11_17
    )

    hm = HistoricalMeanBaseline().fit(train_df)
    hm_val_pred = hm.predict(val_df)
    hm_val_res = eval_val_with_station_slices(y_val, hm_val_pred, val_stations, seen_11_17)
    tuning_log["validation_2018_results"]["historical_mean"] = hm_val_res
    print(f"  HistoricalMean Val 2018: MAE={hm_val_res['overall']['mae']:.4f}, RMSE={hm_val_res['overall']['rmse']:.4f}, R2={hm_val_res['overall']['r2']:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Seasonal Naive Baseline (lag_24h & lag_168h blend tuned on Val 2018)
    # -------------------------------------------------------------------------
    print(f"[{horizon_tag}] Tuning Baseline 2: Seasonal Naive (24h / 168h)...", flush=True)
    sn_trials = []
    best_sn_w = 0.5
    best_sn_rmse = float("inf")
    best_sn_res = None
    for w24 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sn = SeasonalNaiveBaseline(w_24=w24).fit(train_df)
        p = sn.predict(val_df)
        r = eval_val_with_station_slices(y_val, p, val_stations, seen_11_17)
        sn_trials.append({"w_24": w24, "w_168": round(1.0 - w24, 2), "val_overall": r["overall"]})
        if r["overall"]["rmse"] < best_sn_rmse:
            best_sn_rmse = r["overall"]["rmse"]
            best_sn_w = w24
            best_sn_res = r
    tuning_log["seasonal_naive_trials"] = sn_trials
    tuning_log["best_seasonal_naive_params"] = {"w_24": best_sn_w, "w_168": round(1.0 - best_sn_w, 2)}
    tuning_log["validation_2018_results"]["seasonal_naive"] = best_sn_res
    print(f"  SeasonalNaive (w_24={best_sn_w}) Val 2018: MAE={best_sn_res['overall']['mae']:.4f}, RMSE={best_sn_res['overall']['rmse']:.4f}, R2={best_sn_res['overall']['r2']:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # 3. Random Forest Regressor (tuned on Train 2011-2017 -> Val 2018)
    # -------------------------------------------------------------------------
    print(f"[{horizon_tag}] Tuning Model 3: Random Forest Regressor...", flush=True)
    rf_candidates = [
        {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 30, "max_features": 0.5, "max_samples": 0.35},
        {"n_estimators": 120, "max_depth": 16, "min_samples_leaf": 30, "max_features": 0.5, "max_samples": 0.35},
        {"n_estimators": 120, "max_depth": 16, "min_samples_leaf": 60, "max_features": 0.5, "max_samples": 0.35},
        {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 50, "max_features": "sqrt", "max_samples": 0.35},
    ]
    rf_trials = []
    best_rf_params = None
    best_rf_rmse = float("inf")
    best_rf_res = None
    for cfg in rf_candidates:
        t0 = time.time()
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **cfg)
        rf.fit(X_train_imp, y_train)
        p_val = np.clip(rf.predict(X_val_imp), 0.0, None)
        r = eval_val_with_station_slices(y_val, p_val, val_stations, seen_11_17)
        dt = round(time.time() - t0, 1)
        rf_trials.append({"params": cfg, "train_sec": dt, "val_overall": r["overall"]})
        print(f"  RF {cfg} ({dt}s) -> Val RMSE={r['overall']['rmse']:.5f}, MAE={r['overall']['mae']:.5f}, R2={r['overall']['r2']:.5f}", flush=True)
        if r["overall"]["rmse"] < best_rf_rmse:
            best_rf_rmse = r["overall"]["rmse"]
            best_rf_params = cfg
            best_rf_res = r
    tuning_log["random_forest_trials"] = rf_trials
    tuning_log["best_random_forest_params"] = best_rf_params
    tuning_log["validation_2018_results"]["random_forest"] = best_rf_res

    # -------------------------------------------------------------------------
    # 4. XGBoost Regressor (tuned on Train 2011-2017 -> Val 2018 with early stopping)
    # -------------------------------------------------------------------------
    print(f"[{horizon_tag}] Tuning Model 4: XGBoost Regressor...", flush=True)
    xgb_candidates = [
        {"objective": "reg:squarederror", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"objective": "reg:squarederror", "max_depth": 6, "learning_rate": 0.04, "min_child_weight": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"objective": "reg:squarederror", "max_depth": 8, "learning_rate": 0.03, "min_child_weight": 60, "subsample": 0.8, "colsample_bytree": 0.75, "reg_lambda": 5.0},
        {"objective": "count:poisson",    "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"objective": "count:poisson",    "max_depth": 6, "learning_rate": 0.04, "min_child_weight": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"objective": "count:poisson",    "max_depth": 8, "learning_rate": 0.03, "min_child_weight": 60, "subsample": 0.8, "colsample_bytree": 0.75, "reg_lambda": 5.0},
    ]
    xgb_trials = []
    best_xgb_params = None
    best_xgb_n_est = 300
    best_xgb_rmse = float("inf")
    best_xgb_res = None

    for cfg in xgb_candidates:
        t0 = time.time()
        eval_metric = "rmse" if cfg["objective"] == "reg:squarederror" else "poisson-nloglik"
        model = XGBRegressor(
            n_estimators=1000,
            tree_method="hist",
            early_stopping_rounds=40,
            eval_metric=eval_metric,
            random_state=42,
            n_jobs=-1,
            **cfg,
        )
        model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
        best_iter = int(model.best_iteration) + 1
        p_val = np.clip(model.predict(X_val_xgb), 0.0, None)
        r = eval_val_with_station_slices(y_val, p_val, val_stations, seen_11_17)
        dt = round(time.time() - t0, 1)
        xgb_trials.append({
            "params": cfg,
            "best_n_estimators": best_iter,
            "train_sec": dt,
            "val_overall": r["overall"],
        })
        print(f"  XGB obj={cfg['objective']} d={cfg['max_depth']} lr={cfg['learning_rate']} trees={best_iter} ({dt}s) -> Val RMSE={r['overall']['rmse']:.5f}, MAE={r['overall']['mae']:.5f}, R2={r['overall']['r2']:.5f}", flush=True)
        if r["overall"]["rmse"] < best_xgb_rmse:
            best_xgb_rmse = r["overall"]["rmse"]
            best_xgb_params = dict(cfg)
            best_xgb_n_est = best_iter
            best_xgb_res = r

    best_xgb_full_params = dict(best_xgb_params, n_estimators=best_xgb_n_est)
    tuning_log["xgboost_trials"] = xgb_trials
    tuning_log["best_xgboost_params"] = best_xgb_full_params
    tuning_log["validation_2018_results"]["xgboost"] = best_xgb_res

    # -------------------------------------------------------------------------
    # Select Winning Model STRICTLY on Validation 2018 RMSE (never peeking at 2019!)
    # -------------------------------------------------------------------------
    val_ranking = sorted(
        [
            ("historical_mean", hm_val_res["overall"]["rmse"], hm_val_res["overall"]["mae"]),
            ("seasonal_naive", best_sn_res["overall"]["rmse"], best_sn_res["overall"]["mae"]),
            ("random_forest", best_rf_res["overall"]["rmse"], best_rf_res["overall"]["mae"]),
            ("xgboost", best_xgb_res["overall"]["rmse"], best_xgb_res["overall"]["mae"]),
        ],
        key=lambda x: x[1],
    )
    winner_name = val_ranking[0][0]
    tuning_log["selected_winner_on_val_2018"] = winner_name
    tuning_log["val_2018_ranking_by_rmse"] = [
        {"model": m, "val_rmse": rmse, "val_mae": mae} for m, rmse, mae in val_ranking
    ]
    print(f"[{horizon_tag}] WINNER ON VALIDATION 2018: {winner_name} (RMSE={val_ranking[0][1]:.5f}, MAE={val_ranking[0][2]:.5f})", flush=True)

    # -------------------------------------------------------------------------
    # Final Retraining on 2011-2018 (Train + Val) for Untouched 2019 & 2020 Eval
    # -------------------------------------------------------------------------
    print(f"[{horizon_tag}] Retraining all 4 models on 2011-2018 (n={len(train_val_df):,})...", flush=True)
    prep_final = TabularPreprocessor(feature_cols).fit(train_val_df)
    X_tv_imp = prep_final.transform(train_val_df, impute=True)
    X_tv_xgb = prep_final.transform(train_val_df, impute=False)

    # Save Baseline 1 (Historical Mean)
    hm_final = HistoricalMeanBaseline().fit(train_val_df)
    joblib.dump(
        {"model_name": "historical_mean", "horizon": horizon_tag, "model": hm_final, "feature_cols": feature_cols},
        os.path.join(MODELS_DIR, f"baseline_historical_mean_{horizon_tag}.joblib"),
    )

    # Save Baseline 2 (Seasonal Naive)
    sn_final = SeasonalNaiveBaseline(w_24=best_sn_w).fit(train_val_df)
    joblib.dump(
        {"model_name": "seasonal_naive", "horizon": horizon_tag, "model": sn_final, "params": {"w_24": best_sn_w}, "feature_cols": feature_cols},
        os.path.join(MODELS_DIR, f"baseline_seasonal_naive_{horizon_tag}.joblib"),
    )

    # Save Model 3 (Random Forest — final retrain with 150 trees for variance reduction)
    t0 = time.time()
    rf_final_cfg = dict(best_rf_params)
    rf_final_cfg["n_estimators"] = 150
    rf_final = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_final_cfg)
    rf_final.fit(X_tv_imp, y_train_val)
    rf_bundle = {
        "model_name": "random_forest",
        "horizon": horizon_tag,
        "model": rf_final,
        "preprocessor": prep_final,
        "impute_missing": True,
        "feature_cols": feature_cols,
        "params": rf_final_cfg,
        "train_years": "2011-2018",
        "seen_stations": sorted(train_val_df["station_id"].unique().tolist()),
    }
    joblib.dump(rf_bundle, os.path.join(MODELS_DIR, f"random_forest_{horizon_tag}.joblib"), compress=3)
    print(f"  Saved final Random Forest ({round(time.time()-t0, 1)}s).", flush=True)

    # Save Model 4 (XGBoost — final retrain on 2011-2018 using Val-selected n_estimators)
    t0 = time.time()
    xgb_final = XGBRegressor(
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **best_xgb_full_params,
    )
    xgb_final.fit(X_tv_xgb, y_train_val, verbose=False)
    xgb_bundle = {
        "model_name": "xgboost",
        "horizon": horizon_tag,
        "model": xgb_final,
        "preprocessor": prep_final,
        "impute_missing": False,
        "feature_cols": feature_cols,
        "params": best_xgb_full_params,
        "train_years": "2011-2018",
        "seen_stations": sorted(train_val_df["station_id"].unique().tolist()),
    }
    joblib.dump(xgb_bundle, os.path.join(MODELS_DIR, f"xgboost_{horizon_tag}.joblib"), compress=3)
    print(f"  Saved final XGBoost ({round(time.time()-t0, 1)}s).", flush=True)

    # Save alias for best model selected strictly on Val 2018
    winner_map = {
        "historical_mean": os.path.join(MODELS_DIR, f"baseline_historical_mean_{horizon_tag}.joblib"),
        "seasonal_naive": os.path.join(MODELS_DIR, f"baseline_seasonal_naive_{horizon_tag}.joblib"),
        "random_forest": os.path.join(MODELS_DIR, f"random_forest_{horizon_tag}.joblib"),
        "xgboost": os.path.join(MODELS_DIR, f"xgboost_{horizon_tag}.joblib"),
    }
    best_bundle = joblib.load(winner_map[winner_name])
    joblib.dump(best_bundle, os.path.join(MODELS_DIR, f"best_model_{horizon_tag}.joblib"), compress=3)
    print(f"  Saved best_model_{horizon_tag}.joblib -> {winner_name}.", flush=True)

    return tuning_log


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Verifying upstream dataset hashes before training...", flush=True)
    raw_sha_before = sha256_file(RAW_CSV)
    clean_sha_before = sha256_file(CLEANED_CSV)
    feat_sha_before = sha256_file(FEATURES_CSV)

    print(f"Loading Phase 4 feature dataset from {FEATURES_CSV}...", flush=True)
    df = load_hourly_features(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.", flush=True)

    # Verify 24h-ahead zero-leakage perturbation proof before constructing 24h features
    print("Running 24-hour-ahead adversarial anti-leakage perturbation proof...", flush=True)
    assert verify_24h_no_future_leakage(df), "24h-ahead leakage check failed!"
    print("  24-hour-ahead anti-leakage proof PASSED.", flush=True)

    # Run Primary Experiment: 1-Hour-Ahead
    log_1h = run_horizon_training(df, FEATURES_1H, horizon_tag="1h")

    # Add strictly <= t - 24h causal features in memory & run Secondary Experiment: 24-Hour-Ahead
    print("\nConstructing strictly <= t-24h features in memory for Secondary Experiment...", flush=True)
    df_24h = add_24h_ahead_features(df)
    log_24h = run_horizon_training(df_24h, FEATURES_24H, horizon_tag="24h")

    # Verify NO upstream CSVs were altered
    assert sha256_file(RAW_CSV) == raw_sha_before, "Raw CSV hash changed!"
    assert sha256_file(CLEANED_CSV) == clean_sha_before, "Cleaned CSV hash changed!"
    assert sha256_file(FEATURES_CSV) == feat_sha_before, "Features CSV hash changed!"

    meta = {
        "upstream_files_unchanged": True,
        "anti_leakage_24h_perturbation_proof": True,
        "primary_1h_ahead": log_1h,
        "secondary_24h_ahead": log_24h,
    }
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTRAINING COMPLETE! Metadata saved to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
