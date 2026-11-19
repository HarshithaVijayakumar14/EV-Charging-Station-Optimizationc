from flask import Blueprint, jsonify, request

from backend.services.recommendation_service import recommendation_service

recommendations_bp = Blueprint("recommendations", __name__)


@recommendations_bp.route("", methods=["GET"])
def get_recommendations():
    """Returns explainable, rule-based EV demand planning recommendations."""
    try:
        persist = request.args.get("persist", "false").lower() == "true"
        station_id = request.args.get("station_id")
        category = request.args.get("category")

        payload = recommendation_service.generate_recommendations(persist=persist)
        recs = payload["recommendations"]

        if station_id:
            recs = [r for r in recs if r["station_id"] == station_id]
        if category:
            recs = [r for r in recs if r["category"].lower() == category.lower()]

        return jsonify(
            {
                "status": "success",
                "disclaimer": payload["disclaimer"],
                "total_recommendations": len(recs),
                "category_counts": payload["category_counts"],
                "recommendations": recs,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
