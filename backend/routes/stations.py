from flask import Blueprint, jsonify, request

from backend.services.data_service import data_service
from ml.station_recommender import haversine_distance

stations_bp = Blueprint("stations", __name__)


@stations_bp.route("", methods=["GET"])
def list_stations():
    """Returns the list of all 46 available stations with summary statistics."""
    try:
        stations = data_service.get_stations_list()
        return jsonify(
            {
                "status": "success",
                "total_stations": len(stations),
                "stations": stations,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@stations_bp.route("/nearby", methods=["GET"])
def get_nearby_stations():
    """
    Returns stations within a given radius of a target station or GPS coordinates.
    Query params:
      - station_id: str (target station ID)
      OR
      - latitude: float, longitude: float
      - radius_miles: float (default 5.0)
      - limit: int (default 10)
    """
    try:
        st_id = request.args.get("station_id")
        lat_param = request.args.get("latitude")
        lon_param = request.args.get("longitude")
        radius_miles = float(request.args.get("radius_miles", 5.0))
        limit = int(request.args.get("limit", 10))

        origin_lat = None
        origin_lon = None
        origin_name = None

        if st_id:
            st = data_service.get_station_detail(st_id)
            if st is None:
                return jsonify(
                    {
                        "status": "error",
                        "error_code": "STATION_NOT_FOUND",
                        "message": f"Station '{st_id}' was not found.",
                    }
                ), 404
            origin_lat = float(st["latitude"])
            origin_lon = float(st["longitude"])
            origin_name = st.get("station_name", st.get("short_name", st_id))
        elif lat_param is not None and lon_param is not None:
            origin_lat = float(lat_param)
            origin_lon = float(lon_param)
            origin_name = f"Coordinates ({origin_lat:.4f}, {origin_lon:.4f})"
        else:
            return jsonify(
                {
                    "status": "error",
                    "error_code": "MISSING_ORIGIN",
                    "message": "Must provide either station_id or both latitude and longitude query parameters.",
                }
            ), 400

        stations = data_service.get_stations_list()
        nearby = []
        for s in stations:
            if st_id and s["station_id"] == st_id:
                continue
            s_lat = float(s["latitude"])
            s_lon = float(s["longitude"])
            dist = haversine_distance(origin_lat, origin_lon, s_lat, s_lon)
            if dist["miles"] <= radius_miles:
                nearby.append(
                    {
                        "station_id": s["station_id"],
                        "station_name": s.get("station_name", s["short_name"]),
                        "short_name": s["short_name"],
                        "address": s["address"],
                        "latitude": s_lat,
                        "longitude": s_lon,
                        "distance_miles": dist["miles"],
                        "distance_km": dist["km"],
                        "port_types": s.get("port_types", []),
                        "total_sessions": s["total_sessions"],
                    }
                )

        nearby.sort(key=lambda x: x["distance_miles"])
        result = nearby[:limit]

        return jsonify(
            {
                "status": "success",
                "origin": {
                    "name": origin_name,
                    "latitude": origin_lat,
                    "longitude": origin_lon,
                    "radius_miles": radius_miles,
                },
                "count": len(result),
                "nearby_stations": result,
            }
        ), 200
    except ValueError as val_err:
        return jsonify({"status": "error", "message": f"Invalid numerical parameter: {val_err}"}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@stations_bp.route("/<path:station_id>", methods=["GET"])
def get_station(station_id: str):
    """Returns detailed historical statistics and recent series for a selected station."""
    try:
        detail = data_service.get_station_detail(station_id)
        if detail is None:
            return jsonify(
                {
                    "status": "error",
                    "error_code": "STATION_NOT_FOUND",
                    "message": f"Station '{station_id}' was not found in the dataset.",
                    "available_stations": [s["station_id"] for s in data_service.get_stations_list()],
                }
            ), 404
        return jsonify({"status": "success", "station": detail}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
