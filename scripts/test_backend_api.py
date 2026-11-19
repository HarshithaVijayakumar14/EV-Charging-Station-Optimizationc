import json
import os
import sys
from urllib.parse import quote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from backend.app import create_app
from backend.config import CLEANED_CSV, HOURLY_FEATURES_CSV, MODEL_1H_PATH, MODEL_24H_PATH, RAW_CSV
from backend.services.data_service import data_service
from backend.services.prediction_service import prediction_service
from ml.common import sha256_file
from ml.predict_demand import predict_1h_ahead, predict_24h_ahead


def run_tests():
    print("Recording upstream SHA-256 checksums before backend test suite...", flush=True)
    hashes_before = {
        "raw": sha256_file(RAW_CSV),
        "cleaned": sha256_file(CLEANED_CSV),
        "features": sha256_file(HOURLY_FEATURES_CSV),
        "model_1h": sha256_file(MODEL_1H_PATH),
        "model_24h": sha256_file(MODEL_24H_PATH),
    }

    app = create_app()
    client = app.test_client()

    # 1-3. Health endpoint & both models loaded
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health failed: {r.data}"
    h = r.get_json()
    assert h["status"] == "healthy"
    assert h["models"]["model_1h_loaded"] is True
    assert h["models"]["model_24h_loaded"] is True
    assert h["models"]["model_1h_info"]["model_name"] == "xgboost"
    assert h["models"]["model_24h_info"]["model_name"] == "xgboost"
    print("  [PASS] /api/health & both XGBoost models loaded once", flush=True)

    # 4. Dashboard summary
    r = client.get("/api/dashboard/summary")
    assert r.status_code == 200
    s = r.get_json()["summary"]
    assert s["total_sessions"] == 259411
    assert s["station_count"] == 46
    assert abs(s["total_energy_kwh"] - 2216493.71) < 0.1
    assert len(s["top_10_stations"]) == 10
    assert len(s["level_comparison"]) == 2
    print("  [PASS] /api/dashboard/summary matches actual cleaned dataset totals", flush=True)

    # 5. Station list & single station detail (URL-encoded with %2F and %23)
    r = client.get("/api/stations")
    assert r.status_code == 200
    st_list = r.get_json()["stations"]
    assert len(st_list) == 46

    ham2 = "PALO ALTO CA / HAMILTON #2"
    r = client.get(f"/api/stations/{quote(ham2, safe='')}")
    assert r.status_code == 200
    assert r.get_json()["station"]["station_id"] == ham2
    print("  [PASS] /api/stations & /api/stations/<station_id>", flush=True)

    # 6. Invalid station ID returns structured 404
    r = client.get("/api/stations/NON_EXISTENT_STATION_999")
    assert r.status_code == 404
    assert r.get_json()["error_code"] == "STATION_NOT_FOUND"
    r = client.get("/api/predictions/station/NON_EXISTENT_STATION_999")
    assert r.status_code == 404
    r = client.get("/api/predictions/1-hour?station_id=NON_EXISTENT_STATION_999")
    assert r.status_code == 404
    print("  [PASS] Invalid station IDs return clean 404 errors", flush=True)

    # 7. 1-hour & 24-hour predictions (all stations)
    r1 = client.get("/api/predictions/1-hour")
    assert r1.status_code == 200
    d1 = r1.get_json()
    assert d1["forecasted_stations_count"] == 41  # 46 - 5 single-hour Sherman stations
    assert d1["insufficient_history_count"] == 5

    r24 = client.get("/api/predictions/24-hour")
    assert r24.status_code == 200
    d24 = r24.get_json()
    assert d24["forecasted_stations_count"] == 41
    assert d24["insufficient_history_count"] == 5
    assert len(d24["network_hourly_step_totals"]) == 24
    print("  [PASS] /api/predictions/1-hour & /api/predictions/24-hour (41 forecasted, 5 insufficient history)", flush=True)

    # 8. Station-specific predictions & cold-start / insufficient-history handling
    web1 = "PALO ALTO CA / WEBSTER #1"
    rs_ham = client.get(f"/api/predictions/station/{quote(ham2, safe='')}").get_json()
    rs_web = client.get(f"/api/predictions/station/{quote(web1, safe='')}").get_json()
    assert rs_ham["prediction_1h"]["status"] == "ok"
    assert rs_web["prediction_1h"]["status"] == "ok"
    assert len(rs_ham["prediction_24h"]["hourly_predictions"]) == 24
    assert rs_ham["historical_backtest"]["is_historical_backtest"] is True
    # Ensure NO accidental cross-station caching!
    assert rs_ham["prediction_1h"]["station_id"] == ham2
    assert rs_web["prediction_1h"]["station_id"] == web1
    assert rs_ham["prediction_1h"]["predicted_sessions"] != rs_web["prediction_1h"]["predicted_sessions"] or (
        rs_ham["prediction_24h"]["total_predicted_sessions_24h"] != rs_web["prediction_24h"]["total_predicted_sessions_24h"]
    )

    # Cold-start with <24h history (SHERMAN 9 has 1 hour) -> clear insufficient_history response
    sh9 = "PALO ALTO CA / SHERMAN 9"
    rs_sh9 = client.get(f"/api/predictions/station/{quote(sh9, safe='')}").get_json()
    assert rs_sh9["is_cold_start"] is True
    assert rs_sh9["prediction_1h"]["status"] == "insufficient_history"
    assert rs_sh9["prediction_1h"]["predicted_sessions"] is None
    assert rs_sh9["prediction_24h"]["status"] == "insufficient_history"

    # Cold-start with >=24h history (SHERMAN 1 has >168h) -> predicts AND warns is_cold_start=True
    sh1 = "PALO ALTO CA / SHERMAN 1"
    rs_sh1 = client.get(f"/api/predictions/station/{quote(sh1, safe='')}").get_json()
    assert rs_sh1["is_cold_start"] is True
    assert rs_sh1["cold_start_note"] is not None
    assert rs_sh1["prediction_1h"]["status"] == "ok"
    assert rs_sh1["prediction_24h"]["status"] == "ok"
    print("  [PASS] Station-specific predictions, cold-start warnings, insufficient-history safety, no cache bleed", flush=True)

    # 9. Exact numerical parity check with ml/predict_demand.py
    row_1h, _ = prediction_service._build_next_1h_feature_row(ham2)
    direct_1h = float(predict_1h_ahead(pd.DataFrame([row_1h]))[0])
    api_1h = rs_ham["prediction_1h"]["predicted_sessions"]
    assert abs(direct_1h - api_1h) < 1e-4, f"1h parity mismatch: {direct_1h} vs {api_1h}"

    df_24h, _ = prediction_service._build_next_24h_feature_rows(ham2)
    direct_24h = predict_24h_ahead(df_24h)
    api_24h = [x["predicted_sessions"] for x in rs_ham["prediction_24h"]["hourly_predictions"]]
    assert max(abs(float(a) - float(b)) for a, b in zip(direct_24h, api_24h)) < 1e-4, "24h parity mismatch!"
    print("  [PASS] Exact numerical parity verified against ml/predict_demand.py (1h & 24h)", flush=True)

    # 10. History, Recommendations, Model Performance, SQLite history
    rh = client.get("/api/history/hourly")
    assert rh.status_code == 200 and len(rh.get_json()["hourly"]) == 24
    rd = client.get("/api/history/daily?year=2019")
    assert rd.status_code == 200 and len(rd.get_json()["daily"]) == 365

    rrec = client.get("/api/recommendations?persist=true")
    assert rrec.status_code == 200
    rec_json = rrec.get_json()
    assert rec_json["total_recommendations"] > 0
    assert "does not include verified station port capacities" in rec_json["disclaimer"].lower()

    rmp = client.get("/api/model/performance")
    assert rmp.status_code == 200
    mp_json = rmp.get_json()
    assert mp_json["selected_model"]["primary_1h_winner"] == "xgboost"
    assert abs(mp_json["primary_1h"]["test_2019"]["xgboost"]["overall"]["rmse"] - 0.35643) < 1e-5
    assert abs(mp_json["primary_1h"]["ood_2020"]["random_forest"]["overall"]["rmse"] - 0.26614) < 1e-5

    rhist = client.get("/api/predictions/history?limit=10")
    assert rhist.status_code == 200 and rhist.get_json()["count"] > 0
    print("  [PASS] /api/history/*, /api/recommendations, /api/model/performance, & SQLite persistence", flush=True)

    # 11. Re-verify upstream SHA-256 hashes completely unchanged
    hashes_after = {
        "raw": sha256_file(RAW_CSV),
        "cleaned": sha256_file(CLEANED_CSV),
        "features": sha256_file(HOURLY_FEATURES_CSV),
        "model_1h": sha256_file(MODEL_1H_PATH),
        "model_24h": sha256_file(MODEL_24H_PATH),
    }
    assert hashes_before == hashes_after, "CRITICAL: Upstream file modified!"
    print("  [PASS] All upstream datasets & trained models verified 100% unmodified!", flush=True)
    print("\nALL BACKEND TESTS PASSED!", flush=True)


if __name__ == "__main__":
    run_tests()
