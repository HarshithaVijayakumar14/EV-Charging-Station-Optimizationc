import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ml.common import MODELS_DIR, add_24h_ahead_features


def load_model_bundle(horizon: str = "1h", model_path: Optional[str] = None) -> dict:
    """Load a saved model bundle (defaults to Validation-selected best_model_{horizon}.joblib)."""
    if model_path is None:
        if horizon not in ("1h", "24h"):
            raise ValueError("horizon must be '1h' or '24h'")
        model_path = os.path.join(MODELS_DIR, f"best_model_{horizon}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run ml/train_model.py first.")
    return joblib.load(model_path)


def predict_from_dataframe(
    df: pd.DataFrame,
    horizon: str = "1h",
    model_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate non-negative predicted hourly session arrival counts (`sessions_count`)
    from an input feature DataFrame using the saved training-fitted model & preprocessor.
    """
    bundle = load_model_bundle(horizon=horizon, model_path=model_path)
    model_name = bundle["model_name"]
    model = bundle["model"]

    work = df.copy()
    if horizon == "24h":
        required_24h = [c for c in bundle["feature_cols"] if c != "station_encoded"]
        missing_24h = [c for c in required_24h if c not in work.columns]
        if missing_24h:
            if "sessions_count" in work.columns and "energy_kwh_arrival_sum" in work.columns:
                work = add_24h_ahead_features(work)
            else:
                raise ValueError(f"Missing required 24h-ahead columns: {missing_24h}")

    if model_name in ("historical_mean", "seasonal_naive"):
        preds = model.predict(work)
    else:
        prep = bundle["preprocessor"]
        impute = bool(bundle.get("impute_missing", True))
        X = prep.transform(work, impute=impute)
        preds = model.predict(X)

    return np.clip(np.asarray(preds, dtype=np.float64), 0.0, None)


def predict_1h_ahead(df: pd.DataFrame, model_path: Optional[str] = None) -> np.ndarray:
    return predict_from_dataframe(df, horizon="1h", model_path=model_path)


def predict_24h_ahead(df: pd.DataFrame, model_path: Optional[str] = None) -> np.ndarray:
    return predict_from_dataframe(df, horizon="24h", model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Predict hourly EV charging station arrivals (sessions_count).")
    parser.add_argument("--input", required=True, help="Path to input CSV with engineered features.")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV.")
    parser.add_argument("--horizon", choices=["1h", "24h"], default="1h", help="Forecast horizon (1h or 24h).")
    parser.add_argument("--model", default=None, help="Optional explicit path to a .joblib model bundle.")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input, low_memory=False)
    preds = predict_from_dataframe(df_in, horizon=args.horizon, model_path=args.model)

    out_cols = [c for c in ["station_id", "hour_timestamp_pt", "hour_timestamp_utc"] if c in df_in.columns]
    df_out = df_in[out_cols].copy() if out_cols else pd.DataFrame(index=df_in.index)
    df_out[f"predicted_sessions_count_{args.horizon}"] = np.round(preds, 4)
    if "sessions_count" in df_in.columns:
        df_out["actual_sessions_count"] = df_in["sessions_count"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out):,} predictions ({args.horizon}) to {args.output}")


if __name__ == "__main__":
    main()
