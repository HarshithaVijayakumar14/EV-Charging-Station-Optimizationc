from flask import Blueprint, jsonify, request

from backend.database.db import get_recent_prediction_history
from backend.services.data_service import data_service
from backend.services.prediction_service import prediction_service

predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.route("/1-hour", methods=["GET"])
def get_1h_predictions():
    """Generates fresh 1-hour-ahead session arrival predictions for all stations (or a specific ?station_id=...)."""
    try:
        station_id = request.args.get("station_id")
        persist = request.args.get("persist", "true").lower() == "true"

        if station_id:
            if data_service.get_station_detail(station_id) is None:
                return jsonify(
                    {
                        "status": "error",
                        "error_code": "STATION_NOT_FOUND",
                        "message": f"Station '{station_id}' was not found in the dataset.",
                    }
                ), 404
            res = prediction_service.predict_station_1h(station_id, persist=persist)
            return jsonify({"status": "success", "prediction": res}), 200

        res = prediction_service.predict_all_stations_1h(persist=persist)
        return jsonify({"status": "success", **res}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@predictions_bp.route("/24-hour", methods=["GET"])
def get_24h_predictions():
    """Generates fresh 24-hour-ahead session arrival predictions for all stations (or a specific ?station_id=...)."""
    try:
        station_id = request.args.get("station_id")
        persist = request.args.get("persist", "true").lower() == "true"

        if station_id:
            if data_service.get_station_detail(station_id) is None:
                return jsonify(
                    {
                        "status": "error",
                        "error_code": "STATION_NOT_FOUND",
                        "message": f"Station '{station_id}' was not found in the dataset.",
                    }
                ), 404
            res = prediction_service.predict_station_24h(station_id, persist=persist)
            return jsonify({"status": "success", "prediction": res}), 200

        res = prediction_service.predict_all_stations_24h(persist=persist)
        return jsonify({"status": "success", **res}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@predictions_bp.route("/station/<path:station_id>", methods=["GET"])
def get_station_predictions(station_id: str):
    """
    Returns both 1-hour-ahead and 24-hour-ahead forecasts plus recent historical Actual-vs-Predicted
    backtest for a specific station.
    """
    try:
        detail = data_service.get_station_detail(station_id)
        if detail is None:
            return jsonify(
                {
                    "status": "error",
                    "error_code": "STATION_NOT_FOUND",
                    "message": f"Station '{station_id}' was not found in the dataset.",
                }
            ), 404

        persist = request.args.get("persist", "true").lower() == "true"
        backtest_hours = min(max(int(request.args.get("backtest_hours", "72")), 24), 168)

        pred_1h = prediction_service.predict_station_1h(station_id, persist=persist)
        pred_24h = prediction_service.predict_station_24h(station_id, persist=persist)
        backtest = prediction_service.get_station_historical_backtest(station_id, hours=backtest_hours)

        return jsonify(
            {
                "status": "success",
                "station_id": station_id,
                "short_name": detail["short_name"],
                "is_cold_start": detail["is_cold_start"],
                "cold_start_note": detail["cold_start_note"],
                "has_sufficient_history": detail["has_sufficient_history"],
                "prediction_1h": pred_1h,
                "prediction_24h": pred_24h,
                "historical_backtest": backtest,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@predictions_bp.route("/history", methods=["GET"])
def get_prediction_audit_history():
    """Returns logged prediction history stored in SQLite."""
    try:
        station_id = request.args.get("station_id")
        horizon = request.args.get("horizon")
        limit = min(max(int(request.args.get("limit", "50")), 1), 500)
        rows = get_recent_prediction_history(station_id=station_id, horizon=horizon, limit=limit)
        return jsonify({"status": "success", "count": len(rows), "records": rows}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
