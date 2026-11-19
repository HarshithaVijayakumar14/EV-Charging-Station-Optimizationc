"""
backend/routes/availability_calibration.py
Blueprint for Empirical Availability Calibration & Error Propagation Analysis:
- GET /api/availability-calibration/summary
- GET /api/availability-calibration/reliability-diagram
"""

import json
import os
from typing import Dict, Optional
from flask import Blueprint, jsonify, request

availability_calibration_bp = Blueprint("availability_calibration", __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "availability_calibration", "results.json")


def _load_calibration_results() -> Optional[Dict]:
    if not os.path.exists(RESULTS_JSON_PATH):
        return None
    try:
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@availability_calibration_bp.route("/summary", methods=["GET"])
def get_calibration_summary():
    """
    Returns high-level empirical calibration findings, error propagation breakdown,
    distribution shift metrics, dwell assumption test, and model benchmark comparisons.
    """
    data = _load_calibration_results()
    if data is None:
        return jsonify({
            "status": "pending_computation",
            "message": "Availability calibration results have not been generated yet. Run 'python -m ml.availability_calibration --horizon both --seed 42' to compute them."
        }), 200

    return jsonify({
        "status": "success",
        "metadata": data.get("metadata"),
        "dwell_assumption_test": data.get("dwell_assumption_test"),
        "distribution_shift": data.get("distribution_shift"),
        "error_propagation": data.get("error_propagation"),
        "slice_analysis": data.get("slice_analysis"),
        "evaluations_summary": {
            h: {
                split: {
                    m: {
                        "ece": m_data.get("calibration", {}).get("ece"),
                        "brier_score": m_data.get("calibration", {}).get("brier_score"),
                        "brier_reliability": m_data.get("calibration", {}).get("brier_reliability"),
                        "brier_resolution": m_data.get("calibration", {}).get("brier_resolution"),
                        "log_loss": m_data.get("calibration", {}).get("log_loss"),
                        "auroc": m_data.get("calibration", {}).get("auroc"),
                        "auprc": m_data.get("calibration", {}).get("auprc"),
                        "bootstrap_95_ci": m_data.get("calibration", {}).get("bootstrap_95_ci"),
                    }
                    for m, m_data in split_data.items()
                }
                for split, split_data in h_data.items()
            }
            for h, h_data in data.get("evaluations", {}).items()
        }
    }), 200


@availability_calibration_bp.route("/reliability-diagram", methods=["GET"])
def get_reliability_diagram():
    """
    Returns 10-bin reliability diagram points, confidence intervals, and decision-level metrics.
    Query parameters:
        horizon: '1h' (default) or '24h'
        split: '2019_test' (default), '2018_validation', or '2020_ood'
    """
    data = _load_calibration_results()
    if data is None:
        return jsonify({
            "status": "pending_computation",
            "message": "Availability calibration results have not been generated yet."
        }), 200

    horizon = request.args.get("horizon", "1h")
    split = request.args.get("split", "2019_test")

    h_eval = data.get("evaluations", {}).get(horizon, {}).get(split, {})
    diagrams = {}
    for m, m_data in h_eval.items():
        diagrams[m] = {
            "ece": m_data.get("calibration", {}).get("ece"),
            "brier_score": m_data.get("calibration", {}).get("brier_score"),
            "bins": m_data.get("calibration", {}).get("reliability_bins", []),
            "bootstrap_95_ci": m_data.get("calibration", {}).get("bootstrap_95_ci"),
        }

    decision_metrics = data.get("decision_metrics", {}).get(horizon, {}).get(split, {})

    return jsonify({
        "status": "success",
        "horizon": horizon,
        "split": split,
        "diagrams": diagrams,
        "decision_metrics": decision_metrics,
    }), 200
