import os
import sys

# Ensure project root is on sys.path when running `python backend/app.py` directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.config import DEBUG, HOST, PORT
from backend.database.db import get_db_counts, init_db
from backend.routes.charging_assistant import charging_assistant_bp
from backend.routes.dashboard import dashboard_bp
from backend.routes.predictions import predictions_bp
from backend.routes.probabilistic import probabilistic_bp
from backend.routes.recommendations import recommendations_bp
from backend.routes.stations import stations_bp
from backend.services.data_service import data_service
from backend.services.model_service import model_service
from backend.services.probabilistic_service import probabilistic_service


def create_app() -> Flask:
    app = Flask(__name__)
    cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env != "*" else "*"
    CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=True)

    # Initialize SQLite, load Phase 5 trained models ONCE, and warm derived read-only data cache
    init_db()
    model_service.load_all()
    data_service.initialize()
    probabilistic_service.load_all()

    # Register modular route blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(stations_bp, url_prefix="/api/stations")
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(recommendations_bp, url_prefix="/api/recommendations")
    app.register_blueprint(charging_assistant_bp, url_prefix="/api")
    app.register_blueprint(probabilistic_bp, url_prefix="/api/probabilistic")

    @app.route("/api/health", methods=["GET"])
    def health():
        model_health = model_service.get_health_info()
        db_counts = get_db_counts()
        status_code = 200 if model_health["models_ready"] else 503
        return jsonify(
            {
                "status": "healthy" if model_health["models_ready"] else "degraded",
                "service": "EV Charging Station Demand Prediction API",
                "models": model_health,
                "database": db_counts,
            }
        ), status_code

    @app.route("/api/history/hourly", methods=["GET"])
    def history_hourly():
        try:
            return jsonify(
                {
                    "status": "success",
                    "hourly": data_service.get_hourly_history(),
                    "day_of_week": data_service.get_day_of_week_history(),
                }
            ), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/history/daily", methods=["GET"])
    def history_daily():
        try:
            year_param = request.args.get("year")
            limit_param = request.args.get("limit")
            year_val = int(year_param) if year_param else None
            limit_val = int(limit_param) if limit_param else None
            return jsonify(
                {
                    "status": "success",
                    "daily": data_service.get_daily_history(year=year_val, limit=limit_val),
                    "monthly": data_service.get_monthly_history(),
                }
            ), 200
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid integer query parameter for year or limit."}), 400
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/model/performance", methods=["GET"])
    def model_performance():
        """Returns actual saved Phase 5 training metadata and evaluation results (no invented numbers)."""
        try:
            meta = model_service.training_metadata
            ev = model_service.evaluation_results
            if meta is None or ev is None:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Phase 5 model evaluation artifacts are not loaded.",
                    }
                ), 503

            # Return structured actual Phase 5 results cleanly
            def compact_model_dict(section: dict) -> dict:
                out = {}
                for m_name, m_data in section.items():
                    out[m_name] = {
                        "overall": m_data.get("overall"),
                        "zero_arrival_hours": m_data.get("zero_arrival_hours"),
                        "nonzero_arrival_hours": m_data.get("nonzero_arrival_hours"),
                        "weekday": m_data.get("weekday"),
                        "weekend": m_data.get("weekend"),
                        "historical_stations": m_data.get("historical_stations"),
                        "new_stations": m_data.get("new_stations"),
                    }
                return out

            return jsonify(
                {
                    "status": "success",
                    "selected_model": {
                        "primary_1h_winner": meta["primary_1h_ahead"]["selected_winner_on_val_2018"],
                        "secondary_24h_winner": meta["secondary_24h_ahead"]["selected_winner_on_val_2018"],
                        "selection_criterion": "Lowest Validation 2018 RMSE (strictly chronological; 2019 kept untouched until final retrain on 2011-2018)",
                        "ood_2020_note": (
                            "2020 was treated as a separate out-of-distribution (OOD) stress test due to the COVID-19 "
                            "shelter-in-place demand collapse (-87.1% in April 2020 vs. April 2019) and 13 newly opened Sherman stations. "
                            "While XGBoost won Validation 2018, Main Test 2019 (all metrics), cold-start Sherman stations (R²=+0.242), "
                            "and 24h-ahead OOD 2020, Random Forest slightly outperformed the deeper 1h-ahead XGBoost model on overall 2020 OOD RMSE (0.2661 vs. 0.2696)."
                        ),
                    },
                    "splits": {
                        "train": "2011-2017 (707,668 station-hours; 27 stations)",
                        "validation": "2018 (277,536 station-hours; 27 historical + 6 newly opened)",
                        "retrain": "2011-2018 (985,204 station-hours; 33 stations)",
                        "main_test": "2019 (289,080 station-hours; 33 historical stations)",
                        "ood_stress_test": "2020 (258,766 station-hours; 33 historical + 13 new Sherman stations)",
                    },
                    "primary_1h": {
                        "validation_2018": meta["primary_1h_ahead"]["validation_2018_results"],
                        "test_2019": compact_model_dict(ev["primary_1h_ahead"]["test_2019"]),
                        "ood_2020": compact_model_dict(ev["primary_1h_ahead"]["ood_2020"]),
                        "best_xgboost_params": meta["primary_1h_ahead"]["best_xgboost_params"],
                        "best_random_forest_params": meta["primary_1h_ahead"]["best_random_forest_params"],
                        "hourly_calibration_2019_winner": ev["primary_1h_ahead"]["test_2019"]["xgboost"]["by_hour"],
                    },
                    "secondary_24h": {
                        "validation_2018": meta["secondary_24h_ahead"]["validation_2018_results"],
                        "test_2019": compact_model_dict(ev["secondary_24h_ahead"]["test_2019"]),
                        "ood_2020": compact_model_dict(ev["secondary_24h_ahead"]["ood_2020"]),
                        "best_xgboost_params": meta["secondary_24h_ahead"]["best_xgboost_params"],
                        "best_random_forest_params": meta["secondary_24h_ahead"]["best_random_forest_params"],
                    },
                    "feature_importance_1h": ev["feature_importance_1h"]["features_ranked_by_permutation_rmse"][:15],
                    "feature_importance_24h": ev["feature_importance_24h"]["features_ranked_by_permutation_rmse"][:15],
                }
            ), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")

    @app.route("/assets/<path:filename>")
    @app.route("/EV-Charging-Station-Optimizationc/assets/<path:filename>")
    def serve_assets(filename):
        assets_dir = os.path.join(FRONTEND_DIST, "assets")
        if os.path.exists(assets_dir):
            return send_from_directory(assets_dir, filename)
        return jsonify({"status": "error", "message": "Assets directory not found."}), 404

    @app.route("/EV-Charging-Station-Optimizationc", defaults={"path": ""})
    @app.route("/EV-Charging-Station-Optimizationc/<path:path>")
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        if path.startswith("api/"):
            return jsonify({"status": "error", "error_code": "NOT_FOUND", "message": "Requested API endpoint does not exist."}), 404
        if os.path.exists(FRONTEND_DIST):
            file_path = os.path.join(FRONTEND_DIST, path)
            if path != "" and os.path.exists(file_path):
                return send_from_directory(FRONTEND_DIST, path)
            return send_from_directory(FRONTEND_DIST, "index.html")
        return jsonify({
            "service": "EV Charging Demand Prediction API",
            "message": "Frontend build not detected in frontend/dist. Run 'npm run build --prefix frontend' or start Vite dev server on port 5173."
        }), 200

    @app.errorhandler(404)
    def handle_404(_err):
        return jsonify({"status": "error", "error_code": "NOT_FOUND", "message": "Requested endpoint does not exist."}), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
