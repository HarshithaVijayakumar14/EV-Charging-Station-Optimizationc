import os
import sys
import json
import urllib.parse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app

def main():
    app = create_app()
    client = app.test_client()

    data = {}
    print("Exporting health...")
    data["health"] = client.get("/api/health").get_json()

    print("Exporting dashboard summary...")
    data["dashboard_summary"] = client.get("/api/dashboard/summary").get_json()

    print("Exporting hourly history...")
    data["hourly_history"] = client.get("/api/history/hourly").get_json()

    print("Exporting daily history...")
    data["daily_history"] = client.get("/api/history/daily?limit=365").get_json()

    print("Exporting stations list...")
    data["stations"] = client.get("/api/stations").get_json()

    print("Exporting 1-hour predictions...")
    data["pred_1h"] = client.get("/api/predictions/1-hour").get_json()

    print("Exporting 24-hour predictions...")
    data["pred_24h"] = client.get("/api/predictions/24-hour").get_json()

    print("Exporting model performance...")
    data["model_performance"] = client.get("/api/model/performance").get_json()

    print("Exporting probabilistic summary...")
    data["prob_summary"] = client.get("/api/probabilistic/summary").get_json()

    print("Exporting probabilistic calibration...")
    data["prob_calibration"] = client.get("/api/probabilistic/calibration").get_json()

    print("Exporting vehicles catalog...")
    data["vehicles"] = client.get("/api/vehicles").get_json()

    print("Exporting recommendations...")
    data["recommendations"] = client.get("/api/recommendations").get_json()

    print("Exporting availability calibration summary...")
    data["availability_calibration_summary"] = client.get("/api/availability-calibration/summary").get_json()

    print("Exporting reliability diagrams...")
    data["reliability_diagrams"] = {}
    for h in ["1h", "24h"]:
        for s in ["2019_test", "2018_validation", "2020_ood"]:
            diag = client.get(f"/api/availability-calibration/reliability-diagram?horizon={h}&split={s}").get_json()
            if diag:
                data["reliability_diagrams"][f"{h}_{s}"] = diag

    # Pre-cache details and predictions for top 5 stations so explorer & station predictions work offline
    top_stations = [
        "PALO ALTO CA / HAMILTON #2",
        "PALO ALTO CA / BRYANT #2",
        "PALO ALTO CA / HIGH #2",
        "PALO ALTO CA / WEBSTER #2",
        "PALO ALTO CA / RINCONADA LIB 1",
    ]
    data["station_details"] = {}
    data["station_predictions"] = {}
    data["prob_arrivals_by_station"] = {}
    data["prob_availability_by_station"] = {}

    for st_id in top_stations:
        safe_id = urllib.parse.quote(st_id)
        det = client.get(f"/api/stations/{safe_id}").get_json()
        if det:
            data["station_details"][st_id] = det
        
        pred = client.get(f"/api/predictions/station/{safe_id}?backtest_hours=72").get_json()
        if pred:
            data["station_predictions"][st_id] = pred

        arr = client.get(f"/api/probabilistic/arrivals?horizon=1h&station_id={safe_id}").get_json()
        if arr:
            data["prob_arrivals_by_station"][st_id] = arr

        av = client.get(f"/api/probabilistic/availability?horizon=1h&station_id={safe_id}").get_json()
        if av:
            data["prob_availability_by_station"][st_id] = av

    # Sample recommendation payload
    sample_rec_payload = {
        "station_id": "PALO ALTO CA / HAMILTON #2",
        "vehicle_id": "tesla_model_3_sr",
        "current_soc_pct": 20.0,
        "target_soc_pct": 80.0,
        "max_radius_miles": 5.0,
        "max_results": 4
    }
    rec_res = client.post("/api/stations/recommend", json=sample_rec_payload).get_json()
    data["sample_recommendation"] = rec_res

    out_file = os.path.join(PROJECT_ROOT, "frontend", "src", "services", "demoData.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    size_kb = os.path.getsize(out_file) / 1024
    print(f"Successfully exported demo dataset to {out_file} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
