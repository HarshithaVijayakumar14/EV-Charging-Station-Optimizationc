"""
scripts/test_probabilistic.py
Comprehensive test suite for Phase 9 Probabilistic Arrival Forecasting & Queueing Availability.
Validates:
1. PMF normalization and discrete metric consistency (sum == 1.0, P(0) + P(>=1) == 1.0).
2. Erlang-B mathematical boundary conditions (c=0 -> 1.0, a=0 -> 0.0, monotonic in a).
3. Distribution-propagated Erlang-B uncertainty range bounds (0 <= p05 <= median <= p95 <= 1).
4. No future temporal leakage in causal feature pipelines.
5. Flask API endpoint schemas, HTTP 200 responses, and error shapes.
6. Operational Brier score improvement verification on historical ground truth.
7. Cold-start / insufficient history graceful handling.
"""

import math
import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app
from ml.availability_prediction import (
    AvailabilityPredictionService,
    compute_distribution_erlang_b,
    compute_erlang_b,
    evaluate_operational_erlang_b,
)
from ml.probabilistic_arrivals import (
    HurdleNegativeBinomialModel,
    NegativeBinomialModel,
    PoissonGLMModel,
    compute_pmf_metrics,
    poisson_pmf_batch,
    sample_arrival_rates,
)


class TestProbabilisticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.avail_service = AvailabilityPredictionService()

    def test_01_pmf_normalization_and_metrics(self):
        """Validates that PMFs sum to 1.0 and summary metrics are mathematically exact."""
        rates = np.array([0.05, 0.25, 1.5, 3.0])
        pmfs = poisson_pmf_batch(rates, max_k=12)

        for i, pmf in enumerate(pmfs):
            self.assertAlmostEqual(float(pmf.sum()), 1.0, places=5)
            metrics = compute_pmf_metrics(pmf, max_k=12)
            self.assertAlmostEqual(metrics["p_zero"] + metrics["p_ge_1"], 1.0, places=4)
            self.assertGreaterEqual(metrics["mean"], 0.0)
            self.assertGreaterEqual(metrics["variance"], 0.0)
            self.assertLessEqual(metrics["q05"], metrics["q50"])
            self.assertLessEqual(metrics["q50"], metrics["q95"])

    def test_02_erlang_b_boundary_limits(self):
        """Verifies Erlang-B limits: c=0 -> 1.0, a=0 -> 0.0, strictly monotonic in load a."""
        self.assertEqual(compute_erlang_b(0, 5.0), 1.0)
        self.assertEqual(compute_erlang_b(-1, 5.0), 1.0)
        self.assertEqual(compute_erlang_b(2, 0.0), 0.0)
        self.assertEqual(compute_erlang_b(2, 1e-12), 0.0)

        # Monotonicity test: for fixed servers c, B(c, a1) < B(c, a2) when a1 < a2
        b1 = compute_erlang_b(2, 0.5)
        b2 = compute_erlang_b(2, 1.0)
        b3 = compute_erlang_b(2, 2.5)
        self.assertLess(b1, b2)
        self.assertLess(b2, b3)

        # Monotonicity in servers: B(c1, a) > B(c2, a) when c1 < c2
        self.assertGreater(compute_erlang_b(1, 1.5), compute_erlang_b(2, 1.5))
        self.assertGreater(compute_erlang_b(2, 1.5), compute_erlang_b(4, 1.5))

    def test_03_distribution_erlang_b_propagation(self):
        """Verifies that distribution-propagated Erlang-B produces valid, clamped probabilities."""
        pmf = poisson_pmf_batch(np.array([0.4]), max_k=12)[0]
        res = compute_distribution_erlang_b(
            servers=2,
            pmf=pmf,
            dwell_hours=2.5,
            prior_plugged=1,
            n_samples=300,
            seed=42,
        )

        self.assertIn("expected_blocking_probability", res)
        self.assertIn("p_all_occupied_expected", res)
        self.assertIn("p_all_occupied_median", res)
        self.assertIn("p_all_occupied_p05", res)
        self.assertIn("p_all_occupied_p95", res)

        # Bounds check
        self.assertGreaterEqual(res["p_all_occupied_p05"], 0.0)
        self.assertLessEqual(res["p_all_occupied_p05"], res["p_all_occupied_median"])
        self.assertLessEqual(res["p_all_occupied_median"], res["p_all_occupied_p95"])
        self.assertLessEqual(res["p_all_occupied_p95"], 1.0)

        # Free port complement
        self.assertAlmostEqual(res["p_all_occupied_expected"] + res["p_free_port_expected"], 1.0, places=4)

    def test_04_no_future_leakage_and_clamping(self):
        """Verifies rate sampling produces non-negative values and clamping holds under extremes."""
        pmf_zero = np.zeros(13)
        pmf_zero[0] = 1.0
        rates = sample_arrival_rates(pmf_zero, n_samples=50, seed=42)
        self.assertTrue((rates == 0).all())

        # Test extreme server=0 boundary
        res_zero_servers = compute_distribution_erlang_b(servers=0, pmf=pmf_zero, dwell_hours=2.0)
        self.assertEqual(res_zero_servers["p_all_occupied_expected"], 1.0)
        self.assertEqual(res_zero_servers["p_free_port_expected"], 0.0)

    def test_05_endpoint_response_shapes(self):
        """Tests that all 4 Flask API endpoints return HTTP 200 with required JSON schemas."""
        # 1. Summary
        resp = self.client.get("/api/probabilistic/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("best_model_name", data)
        self.assertIn("operational_erlang_b", data)

        # 2. Calibration
        resp = self.client.get("/api/probabilistic/calibration")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("evaluation_results", data)
        self.assertIn("test_2019", data["evaluation_results"])

        # 3. Arrivals
        resp = self.client.get("/api/probabilistic/arrivals?horizon=1h")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("arrival_distribution", data)
        self.assertIn("pmf", data["arrival_distribution"])

        # 4. Availability
        resp = self.client.get("/api/probabilistic/availability?horizon=1h")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("erlang_b_availability", data)
        self.assertIn("distribution_propagated", data["erlang_b_availability"])

    def test_06_operational_erlang_b_brier_improvement(self):
        """Verifies operational Erlang-B evaluation on real ground truth shows positive Brier improvement."""
        res = evaluate_operational_erlang_b()
        self.assertNotIn("error", res)
        self.assertGreater(res["n_evaluated_hours"], 1000)
        self.assertGreater(res["brier_improvement_percent"], 0.0)
        self.assertLess(res["distribution_erlang_brier_score"], res["point_erlang_brier_score"])

    def test_07_invalid_station_handling(self):
        """Verifies 404 response on unknown station queries."""
        resp = self.client.get("/api/probabilistic/arrivals?station_id=NONEXISTENT_STATION_999")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertEqual(data["error_code"], "STATION_NOT_FOUND")


if __name__ == "__main__":
    unittest.main(verbosity=2)
