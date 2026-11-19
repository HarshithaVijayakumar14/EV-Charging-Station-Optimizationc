from flask import Blueprint, jsonify

from backend.database.db import get_db_counts
from backend.services.data_service import data_service
from backend.services.model_service import model_service

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/summary", methods=["GET"])
def get_dashboard_summary():
    """Returns actual historical totals, station counts, average hourly rates, and model availability."""
    try:
        summary = data_service.get_summary()
        model_info = model_service.get_health_info()
        db_counts = get_db_counts()
        return jsonify(
            {
                "status": "success",
                "summary": summary,
                "models": model_info,
                "database": db_counts,
            }
        ), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
