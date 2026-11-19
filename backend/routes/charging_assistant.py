"""
backend/routes/charging_assistant.py
Blueprint for Phase 8 Intelligent EV Charging Assistant:
- Charging point inventory (all points, station-specific)
- Historical availability queries
- Future availability predictions (1h, 24h, or target datetime)
- Vehicle catalog
- Charging time estimation
- Intelligent nearby station recommendations
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from flask import Blueprint, jsonify, request

from backend.database.db import log_availability_request, log_charging_estimate
from backend.services.data_service import data_service
from ml.availability_prediction import availability_service
from ml.charging_point_inventory import inventory_service
from ml.charging_time_estimator import calculate_charging_time
from ml.historical_availability import historical_availability_service
from ml.station_compatibility import check_vehicle_station_compatibility
from ml.station_recommender import station_recommender

charging_assistant_bp = Blueprint("charging_assistant", __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEHICLE_CATALOG_CSV = os.path.join(PROJECT_ROOT, "data", "vehicle_catalog.csv")

_cached_vehicles: Optional[List[Dict]] = None


def get_vehicles_catalog() -> List[Dict]:
    global _cached_vehicles
    if _cached_vehicles is None:
        if os.path.exists(VEHICLE_CATALOG_CSV):
            df = pd.read_csv(VEHICLE_CATALOG_CSV)
            _cached_vehicles = df.to_dict(orient="records")
        else:
            _cached_vehicles = []
    return _cached_vehicles


# ==============================================================================
# 1. Charging Point Inventory Endpoints
# ==============================================================================

@charging_assistant_bp.route("/charging-points/inventory", methods=["GET"])
def get_charging_points_inventory():
    """Returns complete verified charging point inventory across all 46 stations."""
    try:
        points = inventory_service.get_all_points()
        active_count = sum(1 for p in points if p.get("is_active_end_of_data"))
        unique_stations = len({p.get("station_id") for p in points})

        return jsonify(
            {
                "status": "success",
                "total_points": len(points),
                "active_points_count": active_count,
                "historical_upgraded_points_count": len(points) - active_count,
                "unique_stations": unique_stations,
                "inventory": points,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@charging_assistant_bp.route("/charging-points/station/<path:station_id>", methods=["GET"])
def get_station_charging_points(station_id: str):
    """Returns verified charging point hardware inventory for a specific station."""
    try:
        summary = inventory_service.get_station_summary(station_id)
        if summary is None:
            return jsonify(
                {
                    "status": "error",
                    "error_code": "STATION_NOT_FOUND",
                    "message": f"Station '{station_id}' was not found in charging point inventory.",
                }
            ), 404
        return jsonify({"status": "success", "station_inventory": summary}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==============================================================================
# 2. Availability Endpoints (Historical & Future)
# ==============================================================================

@charging_assistant_bp.route("/availability/historical", methods=["GET"])
def get_historical_availability():
    """
    Returns historical availability records filtered by station, date, and hour.
    Query params:
      - station_id: str (optional)
      - date: YYYY-MM-DD (optional)
      - hour: 0-23 (optional)
      - limit: int (optional, default 50, max 200)
    """
    try:
        station_id = request.args.get("station_id")
        date = request.args.get("date")
        hour_param = request.args.get("hour")
        limit_param = request.args.get("limit", "50")

        hour = int(hour_param) if hour_param is not None and hour_param != "" else None
        limit = min(int(limit_param), 200)

        records = historical_availability_service.query_availability(
            station_id=station_id,
            date=date,
            hour=hour,
            limit=limit,
        )

        return jsonify(
            {
                "status": "success",
                "count": len(records),
                "station_id": station_id,
                "records": records,
            }
        ), 200
    except ValueError as val_err:
        return jsonify({"status": "error", "message": f"Invalid integer parameter: {val_err}"}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@charging_assistant_bp.route("/availability/predict", methods=["GET"])
def predict_availability():
    """
    Returns predicted availability metrics for a station.
    Query params:
      - station_id: str (required)
      - horizon: '1h' or '24h' (optional, default '1h')
      - datetime: ISO timestamp string (optional, e.g. 2020-03-15T14:00:00)
    """
    try:
        station_id = request.args.get("station_id")
        if not station_id:
            return jsonify({"status": "error", "message": "Missing required query parameter: station_id"}), 400

        horizon = request.args.get("horizon", "1h").lower()
        target_dt = request.args.get("datetime")

        if target_dt:
            result = availability_service.estimate_at_target_datetime(station_id, target_dt)
        elif horizon == "24h":
            result = availability_service.estimate_station_availability_24h(station_id)
        else:
            result = availability_service.estimate_station_availability_1h(station_id)

        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 404

        return jsonify({"status": "success", "availability_prediction": result}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==============================================================================
# 3. Vehicle Catalog Endpoint
# ==============================================================================

@charging_assistant_bp.route("/vehicles", methods=["GET"])
def list_vehicles():
    """Returns the verified catalog of electric vehicles (four-wheelers, two-wheelers, custom)."""
    try:
        category = request.args.get("category")
        catalog = get_vehicles_catalog()
        if category:
            catalog = [v for v in catalog if v.get("vehicle_category", "").lower() == category.lower()]
        return jsonify(
            {
                "status": "success",
                "total_vehicles": len(catalog),
                "vehicles": catalog,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==============================================================================
# 4. Charging Time Estimation Endpoint
# ==============================================================================

@charging_assistant_bp.route("/charging/estimate", methods=["POST"])
def estimate_charging():
    """
    Calculates charging time estimate taking into account:
    - Battery capacity (kWh)
    - Current SOC % and Target SOC %
    - Station power (kW) and Vehicle onboard max AC power (kW)
    - Charger efficiency (default 90%)
    - CC/CV battery taper slow-down above 80% SOC
    """
    try:
        data = request.get_json(silent=True) or {}
        batt_cap = float(data.get("battery_capacity_kwh", 60.0))
        cur_soc = float(data.get("current_soc_pct", 20.0))
        tgt_soc = float(data.get("target_soc_pct", 80.0))
        veh_max_power = float(data.get("vehicle_max_power_kw", 11.0))
        efficiency = float(data.get("efficiency", 0.90))
        station_id = data.get("station_id")
        station_power = data.get("station_power_kw")

        if tgt_soc <= cur_soc:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Target SOC ({tgt_soc}%) must be greater than current SOC ({cur_soc}%).",
                }
            ), 400

        # Resolve station power if station_id provided
        if station_power is None:
            if station_id:
                st_meta = data_service.get_station_detail(station_id)
                if st_meta and st_meta.get("port_type") == "Level 1":
                    station_power = 1.92
                else:
                    station_power = 6.6
            else:
                station_power = 6.6
        else:
            station_power = float(station_power)

        res = calculate_charging_time(
            battery_capacity_kwh=batt_cap,
            current_soc_pct=cur_soc,
            target_soc_pct=tgt_soc,
            station_power_kw=station_power,
            vehicle_max_power_kw=veh_max_power,
            efficiency=efficiency,
        )

        # Log charging estimate in SQLite
        veh_model = str(data.get("vehicle_model", "Custom / Unspecified"))
        log_charging_estimate(
            vehicle_model=veh_model,
            battery_capacity_kwh=batt_cap,
            required_energy_kwh=res["required_energy_kwh"],
            station_power_kw=station_power,
            estimated_duration_minutes=res["estimated_duration_minutes"],
        )

        res["station_rated_power_kw"] = station_power

        return jsonify({"status": "success", "estimate": res}), 200
    except ValueError as val_err:
        return jsonify({"status": "error", "message": f"Invalid numerical parameter: {val_err}"}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ==============================================================================
# 5. Intelligent Station Recommendation Endpoint
# ==============================================================================

@charging_assistant_bp.route("/stations/recommend", methods=["POST"])
def recommend_stations():
    """
    Intelligently assesses a user-selected target station and recommends nearby alternatives
    based on geodesic distance, vehicle compatibility, predicted availability, and charging speed.
    """
    try:
        data = request.get_json(silent=True) or {}
        station_id = data.get("station_id")
        if not station_id:
            return jsonify({"status": "error", "message": "Missing required field: station_id"}), 400

        target_meta = data_service.get_station_detail(station_id)
        if target_meta is None:
            return jsonify(
                {
                    "status": "error",
                    "error_code": "STATION_NOT_FOUND",
                    "message": f"Target station '{station_id}' was not found.",
                }
            ), 404

        cur_soc = float(data.get("current_soc_pct", 20.0))
        tgt_soc = float(data.get("target_soc_pct", 80.0))
        arrival_iso = data.get("target_datetime_pt")
        max_radius = float(data.get("max_radius_miles", 5.0))
        max_results = int(data.get("max_results", 4))

        # Resolve vehicle specifications
        vehicle = data.get("vehicle")
        vehicle_id = data.get("vehicle_id")

        if vehicle is None and vehicle_id:
            catalog = get_vehicles_catalog()
            match = next((v for v in catalog if v.get("vehicle_id") == vehicle_id), None)
            if match:
                vehicle = match

        if vehicle is None:
            vehicle = {
                "manufacturer": "Generic / Custom",
                "model": "Four-Wheeler Template",
                "vehicle_category": "Four-Wheeler",
                "battery_capacity_kwh": 60.0,
                "maximum_ac_charging_power_kw": 11.0,
                "connector_type": "J1772",
                "charging_efficiency": 0.90,
            }

        # 1. Target station compatibility
        target_compat = check_vehicle_station_compatibility(vehicle, station_id)

        # 2. Target station availability
        if arrival_iso:
            target_avail = availability_service.estimate_at_target_datetime(station_id, arrival_iso)
        else:
            target_avail = availability_service.estimate_station_availability_1h(station_id)

        # 3. Target station charging time
        veh_max_kw = float(vehicle.get("maximum_ac_charging_power_kw", 11.0))
        batt_kwh = float(vehicle.get("battery_capacity_kwh", 60.0))
        eff = float(vehicle.get("charging_efficiency", 0.90))
        st_kw = target_compat["station_max_power_kw"]

        target_chg_time = calculate_charging_time(
            battery_capacity_kwh=batt_kwh,
            current_soc_pct=cur_soc,
            target_soc_pct=tgt_soc,
            station_power_kw=st_kw,
            vehicle_max_power_kw=veh_max_kw,
            efficiency=eff,
        )

        # 4. Find recommended nearby alternatives
        alternatives = station_recommender.recommend_nearby_alternatives(
            target_station_id=station_id,
            vehicle=vehicle,
            target_iso_pt=arrival_iso,
            current_soc_pct=cur_soc,
            target_soc_pct=tgt_soc,
            max_radius_miles=max_radius,
            max_results=max_results,
        )

        # 5. Log user request in SQLite
        est_free = target_avail.get("estimated_free_points")
        p_zero = target_avail.get("probability_zero_free_points")
        avail_status = target_avail.get("availability_status", "Unknown")

        log_availability_request(
            vehicle_model=f"{vehicle.get('manufacturer', '')} {vehicle.get('model', '')}".strip(),
            vehicle_category=vehicle.get("vehicle_category", "Four-Wheeler"),
            current_soc_pct=cur_soc,
            target_soc_pct=tgt_soc,
            station_id=station_id,
            arrival_time_pt=arrival_iso or str(target_avail.get("target_timestamp_pt", "Immediate")),
            estimated_free_points=est_free,
            probability_zero_free=p_zero,
            availability_status=avail_status,
            recommended_alternatives_count=len(alternatives),
        )

        return jsonify(
            {
                "status": "success",
                "target_station": {
                    "station_id": station_id,
                    "station_name": target_meta.get("station_name", target_meta.get("short_name", station_id)),
                    "short_name": target_meta["short_name"],
                    "address": target_meta["address"],
                    "latitude": target_meta["latitude"],
                    "longitude": target_meta["longitude"],
                    "compatibility": target_compat,
                    "availability": target_avail,
                    "charging_time": target_chg_time,
                },
                "recommended_alternatives": alternatives,
                "summary": {
                    "alternatives_found": len(alternatives),
                    "search_radius_miles": max_radius,
                    "target_free_points_estimated": est_free,
                    "target_occupancy_risk": p_zero,
                },
            }
        ), 200
    except ValueError as val_err:
        return jsonify({"status": "error", "message": f"Invalid parameter: {val_err}"}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
