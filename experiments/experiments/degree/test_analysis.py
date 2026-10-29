from copy import deepcopy
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import analysis

FAMILIES = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "holme_kim",
]


def graph_fixture():
    meta = dict(
        graph_id="17",
        family="erdos_renyi",
        num_nodes="6",
        num_edges="6",
        degree_sum="12",
        min_degree="1",
        max_degree="3",
        mean_degree="2.0",
        distinct_degrees="3",
        degrees_sha256="a" * 64,
    )
    prepared = dict(id="17", label=0, num_nodes=6, undirected_edges=6, sha256="b" * 64)
    return (
        meta,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 2, 2], dtype=np.int64),
        prepared,
    )


class BootstrapAndMonteCarloTests(unittest.TestCase):
    def test_precision_produces_exact_trial_count(self):
        for trials in [1, 2, 7, 100, 999, 1000, 2000, 10000]:
            with self.subTest(trials=trials):
                precision = analysis.bootstrap_precision(trials)
                self.assertGreater(precision, 0)
                self.assertEqual(int(0.25 / precision / precision), trials)

    def test_mc_zero_is_not_certainty(self):
        low, high = analysis.mc_interval(0, 1000)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 1000), places=12)
        self.assertGreater(high, 0)

    def test_mc_all_successes_is_not_certainty(self):
        low, high = analysis.mc_interval(1000, 1000)
        self.assertAlmostEqual(low, 0.025 ** (1 / 1000), places=12)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1)

    def test_mc_interval_is_symmetric_at_half(self):
        low, high = analysis.mc_interval(500, 1000)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low, 1 - high, places=12)

    def test_seed_stable_and_graph_specific(self):
        first = [analysis.stable_seed(str(k)) for k in range(20)]
        second = [analysis.stable_seed(str(k)) for k in reversed(range(20))]
        self.assertEqual(first, list(reversed(second)))
        self.assertEqual(len(set(first)), 20)
        self.assertTrue(all((isinstance(seed, int) and seed >= 0 for seed in first)))


class LikelihoodTests(unittest.TestCase):
    def test_exact_discrete_zeta_mass(self):
        k = np.array([1, 2, 3, 10, 100], dtype=np.int64)
        values, method = analysis.powerlaw_logpmf(k, 2.0, 1)
        expected = math.log(6 / math.pi**2) - 2 * np.log(k)
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
        self.assertIsInstance(method, str)
        self.assertTrue(method)

    def test_high_alpha_high_xmin_stays_finite(self):
        k = np.array([19, 20, 21, 25], dtype=np.int64)
        values, _ = analysis.powerlaw_logpmf(k, 17.42, 19)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values <= 0).all())
        np.testing.assert_allclose(
            np.diff(values), -17.42 * np.diff(np.log(k)), atol=1e-12
        )

    def test_underflow_is_explicit_or_recovers_finite_values(self):
        try:
            values, _ = analysis.powerlaw_logpmf(np.array([20, 21, 22]), 1000.0, 20)
        except (ValueError, ArithmeticError):
            return
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values <= 1e-12).all())


class AlternativeFitTests(unittest.TestCase):
    def compare(self, alternate_logs, noise=False, log_pl=None, parameters=None):
        import powerlaw

        alternate_logs = np.array(alternate_logs, dtype=float)
        if log_pl is None:
            log_pl = np.array([-1.0, -2.0, -3.0])
        fitted = SimpleNamespace(
            parameters=parameters or {"Lambda": 0.5},
            in_range=lambda: True,
            noise_flag=noise,
            loglikelihoods=lambda tail: alternate_logs.copy(),
            pdf=lambda tail: np.exp(alternate_logs),
        )
        with patch.object(powerlaw, "Exponential", return_value=fitted):
            return analysis.fit_alternative(
                np.array([2, 3, 4]), np.array(log_pl, dtype=float), 2, "exponential"
            )

    def test_failed_convergence_is_not_reported_as_valid_comparison(self):
        result = self.compare([-2.0, -1.5, -3.1], noise=True)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["noise_flag"])
        self.assertNotIn("preference_at_010", result)

    def test_zero_variance_comparison_is_invalid(self):
        result = self.compare([-1.0, -2.0, -3.0])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("preference_at_010", result)

    def test_library_probability_floor_is_not_a_valid_fit(self):
        result = self.compare([math.log(1e-308), -1.5, -3.1])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("preference_at_010", result)

    def test_lr_sign_and_scale_match_independent_formula(self):
        from scipy.special import erfc

        alternate_logs = np.array([-2.0, -1.5, -3.1])
        log_pl = np.array([-1.0, -2.0, -3.0])
        difference = log_pl - alternate_logs
        expected_r = float(difference.sum())
        expected_scaled = expected_r / math.sqrt(
            len(difference) * float(difference.var())
        )
        expected_p = float(erfc(abs(expected_scaled) / math.sqrt(2)))
        result = self.compare(alternate_logs, log_pl=log_pl)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["log_likelihood_ratio"], expected_r, places=12)
        self.assertAlmostEqual(result["normalized_ratio"], expected_scaled, places=12)
        self.assertAlmostEqual(result["nominal_p"], expected_p, places=12)
        self.assertEqual(result["preference_at_010"], "inconclusive")

    def test_predeclared_point_one_comparison_threshold(self):
        log_pl = np.array([-3.0, -3.0, -3.0])
        differences = np.array([-1.0, 0.0, 1.0]) + 0.825
        result = self.compare(log_pl - differences, log_pl=log_pl)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["nominal_p"], 0.05)
        self.assertLess(result["nominal_p"], 0.1)
        self.assertEqual(result["preference_at_010"], "power_law")

    def test_nonfinite_parameter_failure_still_serializes(self):
        result = self.compare([-2.0, -1.5, -3.1], parameters={"Lambda": float("nan")})
        self.assertEqual(result["status"], "failed")
        json.dumps(result, allow_nan=False)

    def test_nonfinite_variance_failure_still_serializes(self):
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.compare([-2.0, -1.5, -3.1], log_pl=[-1e308, -2.0, -3.0])
        self.assertEqual(result["status"], "failed")
        json.dumps(result, allow_nan=False)


class HistogramValidationTests(unittest.TestCase):
    def call(self, fixture):
        return analysis.validate_graph_histogram(*fixture, FAMILIES)

    def test_valid_histogram_preserves_counts(self):
        record = self.call(graph_fixture())
        self.assertEqual(record["graph_id"], "17")
        self.assertEqual(record["family"], "erdos_renyi")
        self.assertEqual(record["num_nodes"], 6)
        self.assertEqual(record["num_edges"], 6)
        np.testing.assert_array_equal(record["degree_values"], [1, 2, 3])
        np.testing.assert_array_equal(record["degree_counts"], [2, 2, 2])
        self.assertEqual(len(record["histogram_sha256"]), 64)

    def test_rejects_incorrect_count_sum(self):
        m, k, c, p = graph_fixture()
        c[0] += 1
        with self.assertRaises(ValueError):
            self.call((m, k, c, p))

    def test_rejects_wrong_handshake_sum(self):
        m, k, c, p = graph_fixture()
        c = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            self.call((m, k, c, p))

    def test_rejects_duplicate_degree_bins(self):
        m, k, c, p = graph_fixture()
        k = np.array([1, 2, 2, 3])
        c = np.array([2, 1, 1, 2])
        with self.assertRaises(ValueError):
            self.call((m, k, c, p))

    def test_rejects_nonpositive_count(self):
        m, k, c, p = graph_fixture()
        for value in [0, -1]:
            bad = c.copy()
            bad[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.call((m, k, bad, p))

    def test_rejects_noninteger_histogram(self):
        m, k, c, p = graph_fixture()
        k = k.astype(float)
        k[0] = 1.25
        with self.assertRaises(ValueError):
            self.call((m, k, c, p))

    def test_rejects_simple_graph_bound_violation(self):
        m, k, c, p = graph_fixture()
        k[-1] = 6
        with self.assertRaises(ValueError):
            self.call((m, k, c, p))

    def test_rejects_reference_mismatch(self):
        fixture = graph_fixture()
        for field, value in [
            ("id", "18"),
            ("label", 1),
            ("label", -1),
            ("label", 5),
            ("num_nodes", 7),
            ("undirected_edges", 7),
        ]:
            m, k, c, p = deepcopy(fixture)
            p[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.call((m, k, c, p))

    def test_rejects_false_derived_metadata(self):
        fixture = graph_fixture()
        for field, value in [
            ("min_degree", "0"),
            ("max_degree", "4"),
            ("mean_degree", "2.1"),
            ("distinct_degrees", "2"),
            ("degree_sum", "14"),
        ]:
            m, k, c, p = deepcopy(fixture)
            m[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.call((m, k, c, p))


if __name__ == "__main__":
    unittest.main()
