import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app

def main():
    app = create_app()
    client = app.test_client()

    print("--- 1. Health Endpoint Test ---")
    res = client.get("/api/health")
    print("Health Status Code:", res.status_code)
    health = res.get_json()
    print("Health Status:", health.get("status"))
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"

    print("\n--- 2. Frontend HTML Route Test ---")
    html_res = client.get("/")
    print("Root Status Code:", html_res.status_code)
    html = html_res.data.decode("utf-8")
    print("Frontend HTML length:", len(html))
    print("Contains id='root':", 'id="root"' in html)
    print("Contains index asset:", "assets/index-" in html)
    assert html_res.status_code == 200

    print("\n--- 3. Recommendation Assistant API Test ---")
    payload = {
        "station_id": "PALO ALTO CA / BRYANT #2",
        "vehicle": {
            "manufacturer": "Tesla",
            "model": "Model 3 Standard Range Plus",
            "vehicle_category": "Four-Wheeler",
            "battery_capacity_kwh": 54.0,
            "maximum_ac_charging_power_kw": 7.7,
            "connector_type": "J1772 (via OEM Adapter)",
            "charging_efficiency": 0.90
        },
        "current_soc_pct": 20,
        "target_soc_pct": 80,
        "target_datetime_pt": None,
        "max_radius_miles": 5.0,
        "max_results": 4
    }

    rec_res = client.post("/api/stations/recommend", json=payload)
    print("Recommendation Status Code:", rec_res.status_code)
    assert rec_res.status_code == 200
    data = rec_res.get_json()
    print("Recommendation status:", data.get("status"))
    target = data.get("target_station", {})
    print("Target station:", target.get("short_name"))
    avail = target.get("availability", {})
    print(f"Free points estimated: {avail.get('estimated_free_points')} of {avail.get('observed_point_count')}")
    alts = data.get("recommended_alternatives", [])
    print("Alternative stations returned:", len(alts))
    for alt in alts:
        print(f"  -> Alt: {alt.get('short_name')} | Free: {alt.get('estimated_free_points')}/{alt.get('observed_point_count')} | Rank Score: {alt.get('rank_score')}")

    print("\n--- 4. Alternative Station Selection Simulation ---")
    # Simulate user clicking 'Select Station' on an alternative station:
    alt_station_id = alts[0]["station_id"]
    payload["station_id"] = alt_station_id
    alt_res = client.post("/api/stations/recommend", json=payload)
    print(f"Selecting Alternative '{alt_station_id}' status code:", alt_res.status_code)
    alt_data = alt_res.get_json()
    new_target = alt_data.get("target_station", {})
    print("New Target Station after Select:", new_target.get("short_name"))
    print("New Estimated Free Points:", new_target.get("availability", {}).get("estimated_free_points"))
    assert new_target.get("station_id") == alt_station_id, "Target station did not switch properly!"

    print("\nALL UI LINKING AND SELECTION ENGINE TESTS PASSED SUCCESSFULLY! [PASS]")

if __name__ == "__main__":
    main()
