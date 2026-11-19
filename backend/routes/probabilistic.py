"""
backend/routes/probabilistic.py
REST API endpoints for probabilistic arrival forecasting and distribution-propagated availability.
Endpoints:
- GET /api/probabilistic/arrivals?station_id=&horizon=1h|24h
- GET /api/probabilistic/availability?station_id=&horizon=1h|24h
- GET /api/probabilistic/calibration
- GET /api/probabilistic/summary
"""

from flask import Blueprint, jsonify, request
from backend.services.data_service import data_service
from backend.services.probabilistic_service import probabilistic_service

probabilistic_bp = Blueprint("probabilistic", __name__)


@probabilistic_bp.route("/arrivals", methods=["GET"])
def get_arrivals():
    """
    Returns discrete arrival-count PMF, expected arrivals, variance, P(0), P(>=1), P(>=2),
    quantiles (q05, q50, q95), and plain-language summary for a given station.
    """
    try:
        station_id = request.args.get("station_id")
        horizon = request.args.get("horizon", "1h").lower()

        if not station_id:
            # Default to first station in data service
            stations = data_service.get_stations_list()
            station_id = stations[0]["station_id"] if stations else "PALO ALTO CA / HAMILTON #2"

        if data_service.get_station_detail(station_id) is None:
            return jsonify({
                "status": "error",
                "error_code": "STATION_NOT_FOUND",
                "message": f"Station '{station_id}' was not found in the dataset.",
            }), 404

        res = probabilistic_service.get_arrivals(station_id, horizon=horizon)
        status_code = 200 if res.get("status") in ("success", "insufficient_history") else 500
        return jsonify(res), status_code
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@probabilistic_bp.route("/availability", methods=["GET"])
def get_availability():
    """
    Returns point Erlang-B blocking vs distribution-propagated Erlang-B with 5-95% uncertainty range,
    P(at least one free port), dwell parameters, and data limitations.
    """
    try:
        station_id = request.args.get("station_id")
        horizon = request.args.get("horizon", "1h").lower()

        if not station_id:
            stations = data_service.get_stations_list()
            station_id = stations[0]["station_id"] if stations else "PALO ALTO CA / HAMILTON #2"

        if data_service.get_station_detail(station_id) is None:
            return jsonify({
                "status": "error",
                "error_code": "STATION_NOT_FOUND",
                "message": f"Station '{station_id}' was not found in the dataset.",
            }), 404

        res = probabilistic_service.get_availability(station_id, horizon=horizon)
        status_code = 200 if res.get("status") in ("success", "insufficient_history") else 500
        return jsonify(res), status_code
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@probabilistic_bp.route("/calibration", methods=["GET"])
def get_calibration():
    """
    Returns cached scoring-rule results (Log Score, RPS, Brier), PIT histogram data,
    reliability diagrams for P(arrivals >= 1) and P(>= 2), and 90% interval coverage.
    """
    try:
        res = probabilistic_service.get_calibration()
        status_code = 200 if res.get("status") == "success" else 500
        return jsonify(res), status_code
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@probabilistic_bp.route("/summary", methods=["GET"])
def get_summary():
    """
    Returns headline comparison metrics for the dashboard overview:
    Log Score, RPS, zero-hour Brier, PIT uniformity, and operational Erlang-B gain.
    """
    try:
        res = probabilistic_service.get_summary()
        status_code = 200 if res.get("status") == "success" else 500
        return jsonify(res), status_code
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
