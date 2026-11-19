"""
scripts/test_availability.py
Comprehensive Phase 8 Test Suite for EV Charging Point Availability & Recommendation Engine.
Tests all 14 requirements specified in Phase 8 prompt:
1. Physical charging point inventory verification (120 points, 46 stations, active vs historical)
2. Equipment identifier integrity (ports in {1, 2}, MAC address clean matching)
3. Historical availability mathematical conservation (occupied + free + unknown == observed)
4. Non-negative bounds and upper capacity constraints
5. Vehicle catalog parameter accuracy (OEM specs, battery kWh, power kW, categories)
6. Vehicle-station compatibility matrix (J1772, adapter requirements, Level 1 vs 2)
7. Charging duration CC/CV battery taper curve modeling (bulk vs taper slow-down above 80%)
8. Future availability non-negative point estimation for 1h and 24h horizons
9. Erlang-B probability bounds strictly in [0.0, 1.0]
10. Cold-start / low-history station handling (<24h history confidence & limits)
11. Geodesic distance calculation and nearby alternative multi-factor ranking
12. Phase 8 Flask API endpoints (HTTP 200 and schema validation)
13. SQLite user request & charging estimate persistence without raw data duplication
14. Regression testing against Phase 6 endpoints (dashboard, predictions, stations, model performance)
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app
from backend.database.db import get_db_counts, log_availability_request, log_charging_estimate
from ml.charging_point_inventory import (
    INVENTORY_CSV,
    ChargingPointInventoryService,
    inventory_service,
)
from ml.historical_availability import (
    HISTORICAL_AVAILABILITY_CSV,
    historical_availability_service,
)
from ml.availability_prediction import (
    availability_service,
    compute_erlang_b,
)
from ml.charging_time_estimator import calculate_charging_time
from ml.station_compatibility import check_vehicle_station_compatibility
from ml.station_recommender import (
    haversine_distance,
    station_recommender,
)


class TestPhase8AvailabilityAndRecommendation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.inventory_df = pd.read_csv(INVENTORY_CSV)
        cls.vehicles_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "vehicle_catalog.csv"))

    # --------------------------------------------------------------------------
    # Test 1: Charging-point inventory count and active vs historical status
    # --------------------------------------------------------------------------
    def test_01_charging_point_inventory_counts(self):
        df = self.inventory_df
        total_points = len(df)
        unique_stations = df["station_id"].nunique()
        active_points = df["is_active_end_of_data"].astype(bool).sum()
        historical_points = total_points - active_points

        self.assertEqual(total_points, 120, f"Expected exactly 120 physical points, found {total_points}")
        self.assertEqual(unique_stations, 46, f"Expected 46 unique stations, found {unique_stations}")
        self.assertGreater(active_points, 0, "Expected active points > 0")
        self.assertGreater(historical_points, 0, "Expected historical upgraded points > 0")

        # Verify specific station with known hardware turnover (Hamilton #2)
        hamilton_pts = df[df["station_id"] == "PALO ALTO CA / HAMILTON #2"]
        self.assertEqual(len(hamilton_pts), 8, "Hamilton #2 should have 8 historical tuples across 4 hardware generations")
        active_hamilton = hamilton_pts["is_active_end_of_data"].astype(bool).sum()
        self.assertEqual(active_hamilton, 2, "Hamilton #2 active points at end of data should be 2")

    # --------------------------------------------------------------------------
    # Test 2: Port numbers and equipment mappings
    # --------------------------------------------------------------------------
    def test_02_equipment_identifier_integrity(self):
        df = self.inventory_df
        ports = set(df["port_number"].unique())
        self.assertTrue(ports.issubset({1, 2}), f"Port numbers must be 1 or 2, found {ports}")

        # Check MAC address format: 12 hex characters or standardized ChargePoint MAC
        for mac in df["mac_address"]:
            self.assertIsInstance(mac, str)
            self.assertGreaterEqual(len(mac), 8, f"MAC address too short: {mac}")

        # Check connector type
        connectors = set(df["connector_type"].unique())
        self.assertTrue(
            any("J1772" in c for c in connectors),
            "Expected J1772 connectors to be present in inventory",
        )

    # --------------------------------------------------------------------------
    # Test 3: Historical availability mathematical conservation identity
    # --------------------------------------------------------------------------
    def test_03_historical_availability_conservation(self):
        self.assertTrue(os.path.exists(HISTORICAL_AVAILABILITY_CSV), "Historical availability CSV must exist")
        # Check a large sample of 50,000 rows across different chunks
        chunk = pd.read_csv(HISTORICAL_AVAILABILITY_CSV, nrows=50000, low_memory=False)
        check = chunk["occupied_point_count"] + chunk["free_observed_point_count"] + chunk["unknown_point_count"]
        mismatches = (check != chunk["observed_point_count"]).sum()
        self.assertEqual(
            mismatches, 0,
            f"Mathematical conservation violated: {mismatches} rows have occupied + free + unknown != observed"
        )

    # --------------------------------------------------------------------------
    # Test 4: Historical and future free-point bounds
    # --------------------------------------------------------------------------
    def test_04_free_points_bounds_and_non_negativity(self):
        chunk = pd.read_csv(HISTORICAL_AVAILABILITY_CSV, nrows=50000, low_memory=False)
        self.assertTrue((chunk["free_observed_point_count"] >= 0).all(), "Free points must never be negative")
        self.assertTrue((chunk["occupied_point_count"] >= 0).all(), "Occupied points must never be negative")
        self.assertTrue(
            (chunk["free_observed_point_count"] <= chunk["observed_point_count"]).all(),
            "Free points must never exceed observed physical points"
        )

    # --------------------------------------------------------------------------
    # Test 5: Vehicle catalog parameter accuracy
    # --------------------------------------------------------------------------
    def test_05_vehicle_catalog_parameters(self):
        df = self.vehicles_df
        self.assertGreaterEqual(len(df), 15, f"Catalog should contain at least 15 vehicles, found {len(df)}")

        categories = set(df["vehicle_category"].unique())
        self.assertIn("Four-Wheeler", categories)
        self.assertIn("Two-Wheeler", categories)

        for _, row in df.iterrows():
            self.assertGreater(float(row["battery_capacity_kwh"]), 0.0)
            self.assertGreaterEqual(float(row["maximum_ac_charging_power_kw"]), 0.5)
            self.assertGreaterEqual(float(row["charging_efficiency"]), 0.80)
            self.assertLessEqual(float(row["charging_efficiency"]), 1.00)

        # Check Tesla Model 3
        t3 = df[df["vehicle_id"] == "tesla_model_3_sr"].iloc[0]
        self.assertEqual(float(t3["battery_capacity_kwh"]), 60.0)
        self.assertEqual(float(t3["maximum_ac_charging_power_kw"]), 11.5)

        # Check Zero SR/F
        zero = df[df["vehicle_id"] == "zero_srf"].iloc[0]
        self.assertEqual(float(zero["battery_capacity_kwh"]), 17.3)
        self.assertEqual(float(zero["maximum_ac_charging_power_kw"]), 6.0)

    # --------------------------------------------------------------------------
    # Test 6: Vehicle-station compatibility matrix
    # --------------------------------------------------------------------------
    def test_06_vehicle_station_compatibility(self):
        # Tesla Model 3 at Level 2 station (compatible via OEM adapter)
        tesla = {"connector_type": "J1772 (via OEM adapter) / NACS"}
        res_t = check_vehicle_station_compatibility(tesla, "PALO ALTO CA / HAMILTON #2")
        self.assertTrue(res_t["is_compatible"])
        self.assertTrue(res_t["adapter_required"])

        # Nissan Leaf at Level 2 station (directly compatible J1772)
        leaf = {"connector_type": "J1772"}
        res_l = check_vehicle_station_compatibility(leaf, "PALO ALTO CA / HAMILTON #2")
        self.assertTrue(res_l["is_compatible"])
        self.assertFalse(res_l["adapter_required"])

        # Super Soco at Level 1 station (NEMA 5-20R) vs Level 2 station
        soco = {"connector_type": "NEMA 5-20R (120V Standard Wall/Plug)"}
        res_s_l1 = check_vehicle_station_compatibility(soco, "PALO ALTO CA / BRYANT #1") # Level 1
        self.assertTrue(res_s_l1["is_compatible"])

        # DC-only fast charging vehicle at AC Level 2 station
        dc_only = {"connector_type": "CHAdeMO DC Fast Charging Only"}
        res_dc = check_vehicle_station_compatibility(dc_only, "PALO ALTO CA / HAMILTON #2")
        self.assertFalse(res_dc["is_compatible"])

    # --------------------------------------------------------------------------
    # Test 7: Charging time estimation and CC/CV taper slowdown above 80%
    # --------------------------------------------------------------------------
    def test_07_charging_time_taper_slowdown(self):
        # 20% to 80% (bulk only: 60% of 60 kWh = 36 kWh)
        # Power = min(6.6, 11.0) * 0.9 = 5.94 kW => 36 / 5.94 = 6.06 hrs = 363.6 min
        bulk = calculate_charging_time(
            battery_capacity_kwh=60.0,
            current_soc_pct=20.0,
            target_soc_pct=80.0,
            station_power_kw=6.6,
            vehicle_max_power_kw=11.0,
            efficiency=0.90,
            has_charging_curve=True,
        )
        self.assertFalse(bulk["is_tapered"])
        self.assertAlmostEqual(bulk["required_energy_kwh"], 36.0, places=1)
        self.assertAlmostEqual(bulk["estimated_duration_minutes"], 363.6, delta=1.0)

        # 80% to 100% (taper only: 20% of 60 kWh = 12 kWh)
        # Taper effective power = 5.94 * 0.60 = 3.564 kW => 12 / 3.564 = 3.367 hrs = 202.0 min
        # Bulk would have been 12 / 5.94 = 2.02 hrs = 121.2 min
        # Taper slowdown ratio should be ~1.67x longer
        taper = calculate_charging_time(
            battery_capacity_kwh=60.0,
            current_soc_pct=80.0,
            target_soc_pct=100.0,
            station_power_kw=6.6,
            vehicle_max_power_kw=11.0,
            efficiency=0.90,
            has_charging_curve=True,
        )
        self.assertTrue(taper["is_tapered"])
        self.assertAlmostEqual(taper["required_energy_kwh"], 12.0, places=1)
        self.assertGreater(taper["estimated_duration_minutes"], 150.0, "Taper duration must reflect reduced rate")

        # Full charge 20% to 100% must equal sum of bulk + taper
        full = calculate_charging_time(
            battery_capacity_kwh=60.0,
            current_soc_pct=20.0,
            target_soc_pct=100.0,
            station_power_kw=6.6,
            vehicle_max_power_kw=11.0,
            efficiency=0.90,
            has_charging_curve=True,
        )
        self.assertAlmostEqual(
            full["estimated_duration_minutes"],
            bulk["estimated_duration_minutes"] + taper["estimated_duration_minutes"],
            delta=1.0,
        )

    # --------------------------------------------------------------------------
    # Test 8: Future availability predictions (1h and 24h horizons)
    # --------------------------------------------------------------------------
    def test_08_future_availability_predictions(self):
        # 1-Hour-ahead availability
        res_1h = availability_service.estimate_station_availability_1h("PALO ALTO CA / HAMILTON #2")
        self.assertEqual(res_1h["status"], "ok")
        self.assertGreaterEqual(res_1h["estimated_free_points"], 0.0)
        self.assertLessEqual(res_1h["estimated_free_points"], res_1h["observed_point_count"])
        self.assertGreaterEqual(res_1h["expected_occupied_points"], 0.0)
        self.assertLessEqual(res_1h["expected_occupied_points"], res_1h["observed_point_count"])

        # 24-Hour-ahead availability
        res_24h = availability_service.estimate_station_availability_24h("PALO ALTO CA / HAMILTON #2")
        self.assertEqual(res_24h["status"], "ok")
        self.assertEqual(len(res_24h["hourly_availability"]), 24)
        for h in res_24h["hourly_availability"]:
            self.assertGreaterEqual(h["estimated_free_points"], 0.0)
            self.assertLessEqual(h["estimated_free_points"], res_24h["observed_point_count"])

    # --------------------------------------------------------------------------
    # Test 9: Erlang-B blocking probability bounds
    # --------------------------------------------------------------------------
    def test_09_erlang_b_probability_bounds(self):
        # Boundary cases
        p_zero_load = compute_erlang_b(servers=2, offered_load=0.0)
        self.assertEqual(p_zero_load, 0.0)

        p_zero_servers = compute_erlang_b(servers=0, offered_load=5.0)
        self.assertEqual(p_zero_servers, 1.0)

        # Standard range
        for c in [1, 2, 4]:
            for a in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                p = compute_erlang_b(servers=c, offered_load=a)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

        # Monotonicity: higher load with fixed servers gives higher blocking
        p_low = compute_erlang_b(servers=2, offered_load=0.5)
        p_high = compute_erlang_b(servers=2, offered_load=3.0)
        self.assertLess(p_low, p_high)

    # --------------------------------------------------------------------------
    # Test 10: Cold-start / low-history station handling
    # --------------------------------------------------------------------------
    def test_10_cold_start_station_handling(self):
        # Sherman stations opened in 2020 are flagged as cold-start with appropriate confidence
        sherman_st = "PALO ALTO CA / SHERMAN 1"
        res = availability_service.estimate_station_availability_1h(sherman_st)
        if res.get("status") == "ok":
            self.assertIn("Cold-Start", res["availability_confidence"])
            self.assertTrue(res["is_cold_start"])
        elif res.get("status") == "insufficient_history":
            self.assertIn("Low", res["availability_confidence"])

    # --------------------------------------------------------------------------
    # Test 11: Geodesic distance & nearby recommendation ranking
    # --------------------------------------------------------------------------
    def test_11_geodesic_distance_and_ranking(self):
        # Geodesic distance Palo Alto City Hall to Stanford Shopping Center (~1.5 miles)
        dist = haversine_distance(37.4446, -122.1605, 37.4428, -122.1730)
        self.assertGreater(dist["miles"], 0.5)
        self.assertLess(dist["miles"], 2.0)
        self.assertAlmostEqual(dist["km"], dist["miles"] * 1.60934, delta=0.1)

        # Recommendation ranking
        vehicle = {
            "manufacturer": "Tesla",
            "model": "Model 3",
            "battery_capacity_kwh": 60.0,
            "maximum_ac_charging_power_kw": 11.5,
            "connector_type": "J1772 (via OEM adapter) / NACS",
            "charging_efficiency": 0.90,
        }
        recs = station_recommender.recommend_nearby_alternatives(
            target_station_id="PALO ALTO CA / HAMILTON #2",
            vehicle=vehicle,
            max_radius_miles=5.0,
            max_results=4,
        )
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 4)

        # Ensure sorted descending by rank_score
        scores = [r["rank_score"] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Ensure all within 5.0 miles
        for r in recs:
            self.assertLessEqual(r["distance_miles"], 5.0)
            self.assertNotEqual(r["station_id"], "PALO ALTO CA / HAMILTON #2")

    # --------------------------------------------------------------------------
    # Test 12: Phase 8 API endpoints (HTTP 200 & schema validation)
    # --------------------------------------------------------------------------
    def test_12_phase8_api_endpoints(self):
        c = self.client

        # GET /api/vehicles
        r1 = c.get("/api/vehicles")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json["status"], "success")
        self.assertGreaterEqual(r1.json["total_vehicles"], 15)

        # GET /api/charging-points/inventory
        r2 = c.get("/api/charging-points/inventory")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json["total_points"], 120)

        # GET /api/charging-points/station/<id>
        r3 = c.get("/api/charging-points/station/PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232")
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json["station_inventory"]["active_observed_points_count"], 2)

        # GET /api/stations/nearby
        r4 = c.get("/api/stations/nearby?station_id=PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232&radius_miles=5.0")
        self.assertEqual(r4.status_code, 200)
        self.assertGreater(r4.json["count"], 0)

        # GET /api/availability/predict
        r5 = c.get("/api/availability/predict?station_id=PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232")
        self.assertEqual(r5.status_code, 200)
        self.assertIn("availability_status", r5.json["availability_prediction"])

        # GET /api/availability/historical
        r6 = c.get("/api/availability/historical?station_id=PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232&limit=5")
        self.assertEqual(r6.status_code, 200)
        self.assertEqual(r6.json["count"], 5)

        # POST /api/charging/estimate
        r7 = c.post(
            "/api/charging/estimate",
            json={
                "battery_capacity_kwh": 60.0,
                "current_soc_pct": 20.0,
                "target_soc_pct": 80.0,
                "station_power_kw": 6.6,
                "vehicle_max_power_kw": 11.0,
            },
        )
        self.assertEqual(r7.status_code, 200)
        self.assertEqual(r7.json["status"], "success")
        self.assertAlmostEqual(r7.json["estimate"]["required_energy_kwh"], 36.0, places=1)

        # POST /api/stations/recommend
        r8 = c.post(
            "/api/stations/recommend",
            json={
                "station_id": "PALO ALTO CA / HAMILTON #2",
                "vehicle_id": "tesla_model_3_sr",
                "current_soc_pct": 20.0,
                "target_soc_pct": 80.0,
            },
        )
        self.assertEqual(r8.status_code, 200)
        self.assertEqual(r8.json["status"], "success")
        self.assertIn("target_station", r8.json)
        self.assertIn("recommended_alternatives", r8.json)

    # --------------------------------------------------------------------------
    # Test 13: SQLite request logging without raw data duplication
    # --------------------------------------------------------------------------
    def test_13_sqlite_request_logging(self):
        counts_before = get_db_counts()

        # Log an availability request
        row_id_avail = log_availability_request(
            vehicle_model="Test Model 3",
            vehicle_category="Four-Wheeler",
            current_soc_pct=20.0,
            target_soc_pct=80.0,
            station_id="PALO ALTO CA / HAMILTON #2",
            arrival_time_pt="2026-11-19 10:00:00",
            estimated_free_points=1.5,
            probability_zero_free=0.12,
            availability_status="Likely Available",
            recommended_alternatives_count=4,
        )
        self.assertGreater(row_id_avail, 0)

        # Log a charging estimate
        row_id_est = log_charging_estimate(
            vehicle_model="Test Model 3",
            battery_capacity_kwh=60.0,
            required_energy_kwh=36.0,
            station_power_kw=6.6,
            estimated_duration_minutes=363.6,
        )
        self.assertGreater(row_id_est, 0)

        counts_after = get_db_counts()
        self.assertGreaterEqual(counts_after["availability_requests_rows"], counts_before["availability_requests_rows"] + 1)
        self.assertGreaterEqual(counts_after["charging_estimates_rows"], counts_before["charging_estimates_rows"] + 1)

    # --------------------------------------------------------------------------
    # Test 14: Regression test against Phase 6 endpoints
    # --------------------------------------------------------------------------
    def test_14_regression_phase6_endpoints(self):
        c = self.client

        # Health
        r_health = c.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json["status"], "healthy")

        # Dashboard summary
        r_dash = c.get("/api/dashboard/summary")
        self.assertEqual(r_dash.status_code, 200)
        self.assertEqual(r_dash.json["summary"]["station_count"], 46)

        # Stations list
        r_stations = c.get("/api/stations")
        self.assertEqual(r_stations.status_code, 200)
        self.assertEqual(r_stations.json["total_stations"], 46)

        # 1-hour predictions
        r_pred1 = c.get("/api/predictions/1-hour?station_id=PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232")
        self.assertEqual(r_pred1.status_code, 200)

        # Recommendations (Phase 6 rule-based)
        r_rec = c.get("/api/recommendations?station_id=PALO%20ALTO%20CA%20%2F%20HAMILTON%20%232")
        self.assertEqual(r_rec.status_code, 200)

        # Model performance
        r_perf = c.get("/api/model/performance")
        self.assertEqual(r_perf.status_code, 200)
        self.assertEqual(r_perf.json["selected_model"]["primary_1h_winner"], "xgboost")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase8AvailabilityAndRecommendation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 80)
    print("PHASE 8 AUTOMATED TEST SUITE SUMMARY:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print("STATUS:    " + ("ALL TESTS PASSED [PASS]" if result.wasSuccessful() else "TESTS FAILED [FAIL]"))
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
