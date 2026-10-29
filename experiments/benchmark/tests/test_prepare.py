from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prepare
from lib.common import load_ablation


class FakeData:
    def __init__(self):
        self.node_names = ["Degree", "Pagerank", "K-Core", "Eigenvector", "Closeness"]
        self.graph_names = [
            "Degree Variance",
            "Assortativity",
            "Density",
            "Clustering",
            "Average Path Length",
        ]
        self.class_names = [
            "erdos_renyi",
            "barabasi_albert",
            "watts_strogatz",
            "stochastic_block_model",
            "holme_kim",
        ]
        self.graphs = [{"id": str(i), "label": i % 5} for i in range(50)]
        self.labels = np.asarray([g["label"] for g in self.graphs], dtype=np.int64)
        self.arrays = []
        for i in range(len(self.graphs)):
            n = 3 + i % 3
            x = np.arange(n * 5, dtype=np.float32).reshape(n, 5) + i * 100
            x[:, 4] = 7
            graph = np.arange(5, dtype=np.float32) + i * 1000
            graph[3] = 13
            self.arrays.append(dict(x_topology=x, graph_features=graph))

    def raw(self, index):
        return self.arrays[index]


def protocol():
    from support import load_protocol

    return load_protocol()["main"]


class SelectionTests(unittest.TestCase):
    def settings(self):
        return protocol()["feature_selection"]

    def test_inclusive_pruning_records_five_forests_and_reuses_seeds(self):
        calls = []

        class Forest:
            def __init__(self, **parameters):
                self.parameters = parameters

            def fit(self, X, y):
                calls.append((self.parameters, X.copy(), y.copy()))
                self.feature_importances_ = np.array(
                    [0.05, 0.6, 0.1, 0.2, 0.05] if X.shape[1] == 5 else [0.75, 0.25]
                )
                return self

        X = np.arange(60, dtype=float).reshape(12, 5)
        y = np.arange(12) % 5
        result = prepare.iterative_rf_selection(
            X, y, list("abcde"), 42, "node", self.settings(), Forest
        )
        self.assertEqual(result["selected_indices"], [1, 3])
        self.assertEqual(len(result["passes"]), 2)
        self.assertEqual(len(calls), 10)
        self.assertEqual(
            [c[0]["random_state"] for c in calls[:5]],
            [c[0]["random_state"] for c in calls[5:]],
        )
        for params, fit_x, fit_y in calls[:5]:
            self.assertEqual(
                {k: v for k, v in params.items() if k != "random_state"},
                prepare.RF_PARAMETERS,
            )
            np.testing.assert_array_equal(fit_x, X)
            np.testing.assert_array_equal(fit_y, y)
        for _, fit_x, _ in calls[5:]:
            np.testing.assert_array_equal(fit_x, X[:, [1, 3]])

    def test_importance_ties_use_original_candidate_order(self):
        class Forest:
            def __init__(self, **parameters):
                pass

            def fit(self, X, y):
                self.feature_importances_ = np.full(X.shape[1], 1 / X.shape[1])
                return self

        result = prepare.iterative_rf_selection(
            np.ones((8, 5)),
            np.arange(8) % 2,
            list("abcde"),
            43,
            "node",
            self.settings(),
            Forest,
        )
        self.assertEqual(result["selected_indices"], [0, 1, 2, 3])

    def test_all_zero_importance_retains_every_candidate(self):
        class Forest:
            def __init__(self, **parameters):
                pass

            def fit(self, X, y):
                self.feature_importances_ = np.zeros(X.shape[1])
                return self

        result = prepare.iterative_rf_selection(
            np.ones((8, 5)),
            np.arange(8) % 2,
            list("abcde"),
            42,
            "graph",
            self.settings(),
            Forest,
        )
        self.assertEqual(result["selected_indices"], list(range(5)))
        self.assertFalse(result["passes"][0]["threshold_reached"])
        self.assertEqual(len(result["passes"]), 1)

    def test_invalid_importance_rejected(self):
        class Forest:
            def __init__(self, **parameters):
                pass

            def fit(self, X, y):
                self.feature_importances_ = np.full(X.shape[1], np.nan)
                return self

        with self.assertRaisesRegex(ValueError, "invalid normalized"):
            prepare.iterative_rf_selection(
                np.ones((8, 5)),
                np.arange(8) % 2,
                list("abcde"),
                42,
                "node",
                self.settings(),
                Forest,
            )

    def test_real_forest_reproducibility_and_separate_random_streams(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(80, 5))
        y = (X[:, 0] > 0).astype(int)
        first = prepare.iterative_rf_selection(
            X, y, list("abcde"), 42, "node", self.settings()
        )
        repeated = prepare.iterative_rf_selection(
            X, y, list("abcde"), 42, "node", self.settings()
        )
        graph = prepare.iterative_rf_selection(
            X, y, list("abcde"), 42, "graph", self.settings()
        )
        other_seed = prepare.iterative_rf_selection(
            X, y, list("abcde"), 43, "node", self.settings()
        )
        self.assertEqual(first, repeated)
        self.assertEqual(first["selected_indices"], [0])
        self.assertNotEqual(first["forest_seeds"], graph["forest_seeds"])
        self.assertNotEqual(first["forest_seeds"], other_seed["forest_seeds"])


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ablation = load_ablation()

    def test_split_matches_audited_ablation_and_has_no_overlap(self):
        data = FakeData()
        for seed in [42, 43, 44, 45, 46]:
            split = self.ablation.stratified_split(data, seed, 0.7, 0.15)
            record = prepare.split_record(data, split, seed, "fixture", protocol())
            self.assertEqual(
                record["ids"],
                {k: [str(data.graphs[i]["id"]) for i in v] for k, v in split.items()},
            )
            flat = sum(record["ids"].values(), [])
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(set(flat), set(map(str, range(50))))
            self.assertEqual(record["dataset_fingerprint"], "fixture")

    def test_node_means_use_float64_and_keep_manifest_order(self):
        data = FakeData()
        data.arrays[0]["x_topology"][:, 0] = [100000000.0, 1, -100000000.0]
        means, graph = prepare.collect_graph_features(data)
        self.assertEqual(means.dtype, np.dtype(np.float64))
        self.assertAlmostEqual(means[0, 0], 1 / 3)
        np.testing.assert_array_equal(graph[13], data.arrays[13]["graph_features"])

    def test_descriptor_csv_retains_labels_order_and_float64_values(self):
        data = FakeData()
        means, graph = prepare.collect_graph_features(data)
        means[0, 0] = 1 / 3
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset_descriptors.csv"
            prepare.save_descriptors(output, data, means, graph)
            with output.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(
                    reader.fieldnames,
                    ["graph_id", "label"]
                    + [f"node_mean_{i}" for i in range(5)]
                    + [f"graph_{i}" for i in range(5)],
                )
                rows = list(reader)
        self.assertEqual([r["graph_id"] for r in rows], [str(i) for i in range(50)])
        self.assertEqual([int(r["label"]) for r in rows], data.labels.tolist())
        np.testing.assert_array_equal(
            np.array([[float(r[f"node_mean_{i}"]) for i in range(5)] for r in rows]),
            means,
        )
        np.testing.assert_array_equal(
            np.array([[float(r[f"graph_{i}"]) for i in range(5)] for r in rows]), graph
        )

    def test_all_fits_ignore_validation_test_features_and_labels(self):
        data = FakeData()
        split = self.ablation.stratified_split(data, 42, 0.7, 0.15)
        original = copy.deepcopy(data.arrays)
        calls = []

        def fixed_selector(X, y, names, seed, level, settings):
            calls.append((level, X.copy(), y.copy()))
            columns = [4, 0] if level == "node" else [3, 1]
            return dict(
                selected_indices=columns, selected_names=[names[i] for i in columns]
            )

        means, graph = prepare.collect_graph_features(data)
        with patch.object(
            prepare, "iterative_rf_selection", side_effect=fixed_selector
        ):
            selection, stats, arrays = prepare.prepare_seed(
                data, split, 42, protocol(), self.ablation, means, graph
            )
        train = split["train"]
        np.testing.assert_array_equal(calls[0][1], means[train])
        np.testing.assert_array_equal(calls[1][1], graph[train])
        np.testing.assert_array_equal(calls[0][2], data.labels[train])
        self.assertEqual(
            stats["node"]["count"], sum((len(data.raw(i)["x_topology"]) for i in train))
        )
        self.assertEqual(stats["graph"]["count"], len(train))
        self.assertEqual(stats["svm"]["count"], len(train))
        self.assertEqual(stats["node"]["scale"][4], 1)
        self.assertEqual(stats["graph"]["scale"][3], 1)
        self.assertEqual(selection["svm_selected_indices"], [4, 0, 8, 6])
        self.assertEqual(arrays["X"].shape, (50, 4))
        raw = np.concatenate((means, graph), axis=1)[:, [4, 0, 8, 6]]
        expected = (raw - raw[train].mean(axis=0)) / np.where(
            raw[train].std(axis=0) <= 1e-12, 1, raw[train].std(axis=0)
        )
        np.testing.assert_allclose(arrays["X"], expected)
        for before, after in zip(original, data.arrays):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
        held_out = split["validation"] + split["test"]
        for i in held_out:
            data.arrays[i]["x_topology"] += 999999
            data.arrays[i]["graph_features"] -= 888888
            data.labels[i] = 99
        changed_means, changed_graph = prepare.collect_graph_features(data)
        with patch.object(
            prepare, "iterative_rf_selection", side_effect=fixed_selector
        ):
            changed_selection, changed_stats, changed_arrays = prepare.prepare_seed(
                data, split, 42, protocol(), self.ablation, changed_means, changed_graph
            )
        self.assertEqual(selection, changed_selection)
        self.assertEqual(stats, changed_stats)
        np.testing.assert_array_equal(arrays["X"][train], changed_arrays["X"][train])
        self.assertFalse(
            np.array_equal(arrays["X"][held_out], changed_arrays["X"][held_out])
        )
        for old, new in zip(calls[:2], calls[2:]):
            np.testing.assert_array_equal(old[1], new[1])
            np.testing.assert_array_equal(old[2], new[2])


if __name__ == "__main__":
    unittest.main()
