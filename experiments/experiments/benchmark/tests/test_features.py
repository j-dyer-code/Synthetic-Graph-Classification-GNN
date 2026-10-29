from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"Unit checks of the intervention definitions; no research results are produced."
import unittest
import numpy as np
from lib.features import CONDITIONS, make_features, stable_seed


class FeatureInterventions(unittest.TestCase):
    def setUp(self):
        self.z = np.arange(60, dtype=np.float32).reshape(12, 5) / 13
        self.c = np.repeat(np.arange(4), 3)

    def f(self, condition, seed=42, z=None):
        return make_features(
            self.z if z is None else z, self.c, condition, seed, "graph_12"
        )

    def test_shape_finiteness_and_reproducibility(self):
        for mode in CONDITIONS:
            a = self.f(mode)
            self.assertEqual(a.shape, self.z.shape)
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, self.f(mode))

    def test_constant_has_no_descriptor_values(self):
        np.testing.assert_array_equal(self.f("constant"), np.ones_like(self.z))

    def test_independent_does_not_read_topology_values(self):
        np.testing.assert_array_equal(
            self.f("independent"), self.f("independent", z=self.z * 999 - 21)
        )
        self.assertFalse(
            np.array_equal(self.f("independent"), self.f("independent", seed=43))
        )

    def test_permutation_preserves_entire_rows(self):
        got = self.f("permuted")
        self.assertEqual(sorted(map(tuple, got)), sorted(map(tuple, self.z)))
        self.assertFalse(np.array_equal(got, self.z))

    def test_mixtures_pair_the_same_noise(self):
        e = self.f("independent")
        for mode, a in [
            ("mixture_025", 0.25),
            ("mixture_050", 0.5),
            ("mixture_075", 0.75),
        ]:
            np.testing.assert_allclose(
                self.f(mode),
                np.sqrt(a) * self.z + np.sqrt(1 - a) * e,
                rtol=2e-06,
                atol=1e-06,
            )

    def test_community_shared_component(self):
        centres = self.f("community") * np.sqrt(2) - self.f("independent")
        for k in np.unique(self.c):
            v = centres[self.c == k]
            np.testing.assert_allclose(v, np.broadcast_to(v[0], v.shape), atol=1e-06)
        np.testing.assert_array_equal(
            self.f("community"), self.f("community", z=self.z * 50)
        )

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.f("unrecognised")
        with self.assertRaises(ValueError):
            self.f("topology", z=np.full((12, 5), np.nan))
        with self.assertRaises(ValueError):
            make_features(self.z, self.c[:-1], "community", 42, "g")

    def test_streams_are_distinct(self):
        self.assertNotEqual(
            stable_seed(42, "g", "node_noise"),
            stable_seed(42, "g", "community_centres"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
