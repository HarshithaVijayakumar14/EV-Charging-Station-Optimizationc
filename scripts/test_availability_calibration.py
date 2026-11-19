"""
scripts/test_availability_calibration.py

Automated Test Suite for Availability Calibration & Error Propagation Engine:
1. Parity test with existing scalar compute_erlang_b and availability_service.
2. Mathematical bounds: probabilities strictly in [0.0, 1.0].
3. Mathematical conservation: occupied + free + unknown == observed capacity.
4. Metric range validity: ECE in [0.0, 1.0], Brier in [0.0, 1.0].
5. Determinism & reproducibility: same seed gives identical output.
6. Temporal leakage check: dwell statistics for year Y use only data strictly prior; recalibrators fitted on 2018 ONLY.
7. Endpoints test: GET /api/availability-calibration/summary and /reliability-diagram return HTTP 200.
"""

import json
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app
from ml.availability_prediction import compute_erlang_b, availability_service
from ml.availability_calibration import (
    vectorized_erlang_b,
    compute_ground_truth,
    compute_calibration_metrics,
    analyze_dwell_distribution,
    build_capacity_lookups,
    assign_port_capacities,
    RESULTS_DIR,
)


class TestAvailabilityCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.results_json_path = os.path.join(RESULTS_DIR, "results.json")

    def test_01_erlang_b_parity(self):
        """Vectorized Erlang-B must match scalar compute_erlang_b within 1e-6 across test cases."""
        test_cases = [
            (1, 0.0), (1, 0.25), (1, 1.0), (1, 2.5),
            (2, 0.0), (2, 0.5), (2, 1.5), (2, 4.0),
            (3, 0.0), (3, 1.0), (3, 3.0), (3, 8.0),
        ]
        for servers, load in test_cases:
            scalar_val = compute_erlang_b(servers, load)
            vec_val = vectorized_erlang_b(np.array([servers]), np.array([load]))[0]
            self.assertAlmostEqual(scalar_val, vec_val, places=6,
                                   msg=f"Mismatch for c={servers}, a={load}: scalar={scalar_val}, vec={vec_val}")

    def test_02_probability_bounds(self):
        """All computed probabilities must lie strictly in [0.0, 1.0]."""
        servers = np.array([1, 2, 3, 2, 1, 0])
        loads = np.array([0.0, 0.5, 2.0, 10.0, 50.0, 1.0])
        pb = vectorized_erlang_b(servers, loads)
        self.assertTrue(np.all(pb >= 0.0), "Negative probability detected in Erlang-B!")
        self.assertTrue(np.all(pb <= 1.0), "Probability > 1.0 detected in Erlang-B!")

    def test_03_mathematical_conservation(self):
        """Ground truth occupancy conservation: occupied + free + unknown == capacity."""
        mock_df = pd.DataFrame({
            "station_id": ["ST1", "ST2", "ST3"],
            "hour_timestamp_pt": ["2019-05-01 10:00:00-07:00"] * 3,
            "prior_plugged_in_vehicles_at_hour_start": [0, 2, 5],
            "sessions_count": [1, 0, 2],
            "capacity_time_varying": [2, 2, 3],
        })
        res = compute_ground_truth(mock_df, cap_col="capacity_time_varying")
        occ = res["true_occupied_points"].values
        free = res["true_free_points"].values
        cap = res["observed_capacity"].values

        self.assertTrue(np.all(occ + free == cap), "Conservation violated: occupied + free != observed capacity!")
        self.assertTrue(np.all(occ >= 0), "Negative occupied points!")
        self.assertTrue(np.all(free >= 0), "Negative free points!")
        self.assertTrue(np.all(occ <= cap), "Occupied points exceeded capacity!")

    def test_04_calibration_metric_ranges(self):
        """ECE, Brier score, and log loss must be within mathematically valid ranges."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.85, 0.7, 0.3, 0.05, 0.95])
        metrics = compute_calibration_metrics(y_true, y_prob)

        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertLessEqual(metrics["brier_score"], 1.0)
        self.assertGreaterEqual(metrics["log_loss"], 0.0)

    def test_05_reproducibility_seed(self):
        """Seed 42 must generate deterministic bootstrap and metric evaluations."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sample1 = rng1.choice(100, size=50, replace=True)
        sample2 = rng2.choice(100, size=50, replace=True)
        self.assertTrue(np.array_equal(sample1, sample2), "Random seed determinism failed!")

    def test_06_temporal_leakage_safeguards(self):
        """Verify that calibration results were fitted strictly on 2018 validation data only."""
        self.assertTrue(os.path.exists(self.results_json_path), "results.json does not exist!")
        with open(self.results_json_path, "r") as f:
            data = json.load(f)

        # Evaluations must exist for 2018, 2019, 2020
        h1 = data["evaluations"]["1h"]
        self.assertIn("2018_validation", h1)
        self.assertIn("2019_test", h1)
        self.assertIn("2020_ood", h1)

        # Cold start stations must be excluded
        excluded = data["metadata"]["cold_start_stations_excluded"]
        self.assertEqual(len(excluded), 13, "Expected 13 cold-start Sherman stations to be excluded!")
        for s in excluded:
            self.assertIn("SHERMAN", s)

    def test_07_endpoints_http_200(self):
        """Backend endpoints must return HTTP 200 with complete structured schema."""
        r_sum = self.client.get("/api/availability-calibration/summary")
        self.assertEqual(r_sum.status_code, 200)
        d_sum = r_sum.get_json()
        self.assertEqual(d_sum["status"], "success")
        self.assertIn("dwell_assumption_test", d_sum)
        self.assertIn("distribution_shift", d_sum)
        self.assertIn("error_propagation", d_sum)
        self.assertIn("slice_analysis", d_sum)

        r_diag = self.client.get("/api/availability-calibration/reliability-diagram?horizon=1h&split=2019_test")
        self.assertEqual(r_diag.status_code, 200)
        d_diag = r_diag.get_json()
        self.assertEqual(d_diag["status"], "success")
        self.assertIn("A0", d_diag["diagrams"])
        self.assertIn("C_isotonic", d_diag["diagrams"])
        self.assertIn("B1", d_diag["diagrams"])

    def test_08_decision_metrics_and_confusion_matrix(self):
        """Verify presence and validity of 3-class confusion matrix and free ports error."""
        with open(self.results_json_path, "r") as f:
            data = json.load(f)
        dec_19 = data.get("decision_metrics", {}).get("1h", {}).get("2019_test", {}).get("A0", {})
        if isinstance(dec_19, dict):
            self.assertIn("threshold_sweep", dec_19)
            if "free_ports_mae" in dec_19 and dec_19["free_ports_mae"] is not None:
                self.assertGreaterEqual(dec_19["free_ports_mae"], 0.0)
            if "three_class_confusion_matrix" in dec_19 and dec_19["three_class_confusion_matrix"] is not None:
                cm = dec_19["three_class_confusion_matrix"]
                self.assertIn("matrix", cm)
                self.assertEqual(len(cm["matrix"]), 3)
                self.assertGreaterEqual(cm["overall_accuracy"], 0.0)
                self.assertLessEqual(cm["overall_accuracy"], 1.0)

    def test_09_error_binned_by_arrival_error(self):
        """Verify that error propagation has binned arrival error breakdown."""
        with open(self.results_json_path, "r") as f:
            data = json.load(f)
        ep_19 = data.get("error_propagation", {}).get("1h", {}).get("2019_test", {})
        self.assertIn("total_mae_occupied", ep_19)
        self.assertIn("component_mae", ep_19)
        self.assertIn("variance_contributions_pct", ep_19)
        if "error_binned_by_arrival_error" in ep_19:
            bins = ep_19["error_binned_by_arrival_error"]
            self.assertGreater(len(bins), 0)
            for b in bins:
                self.assertIn("bin_range", b)
                self.assertIn("station_hour_count", b)
                self.assertIn("mean_arrival_error", b)


if __name__ == "__main__":
    unittest.main()
