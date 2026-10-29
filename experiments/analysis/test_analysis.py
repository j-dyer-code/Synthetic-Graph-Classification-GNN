from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import t
import metrics
from result_io import prediction_frame, verified_sources, verify_prepared


class PairedIntervalTests(unittest.TestCase):
    def test_all_36_pairs_and_declared_critical_value(self):
        runs = []
        for model_index, architecture in enumerate(metrics.MODELS):
            for seed_index, seed in enumerate(metrics.SEEDS):
                accuracy = (
                    0.7
                    + model_index * 0.01
                    + (model_index + 1) * (seed_index - 2) * 0.0007
                )
                runs.append(
                    SimpleNamespace(
                        seed=seed,
                        architecture=architecture,
                        aggregate={"accuracy": accuracy},
                    )
                )
        intervals, differences = metrics.paired_accuracy_intervals(runs)
        self.assertEqual(len(intervals), 72)
        self.assertEqual(len(differences), 180)
        self.assertEqual(set(intervals.family_size), {36})
        lookup = {(r.seed, r.architecture): r.aggregate["accuracy"] for r in runs}
        for row in intervals.itertuples():
            d = [
                lookup[seed, row.architecture_a] - lookup[seed, row.architecture_b]
                for seed in metrics.SEEDS
            ]
            mean = sum(d) / 5
            variance = sum(((x - mean) ** 2 for x in d)) / 4
            se = math.sqrt((1 / 5 + 300 / row.n_denominator) * variance)
            width = float(t.ppf(1 - 0.05 / 72, 4)) * se
            self.assertAlmostEqual(row.corrected_standard_error, se, places=14)
            self.assertAlmostEqual(row.bonferroni95_low, mean - width, places=14)
            self.assertAlmostEqual(row.bonferroni95_high, mean + width, places=14)

    def test_zero_variance_missing_and_duplicate_runs(self):
        runs = [
            SimpleNamespace(seed=seed, architecture=model, aggregate={"accuracy": 0.75})
            for seed in metrics.SEEDS
            for model in metrics.MODELS
        ]
        frame, _ = metrics.paired_accuracy_intervals(runs)
        self.assertTrue(
            (frame.interval_status == "not_estimable_zero_or_roundoff_variance").all()
        )
        self.assertTrue(frame.bonferroni95_low.isna().all())
        with self.assertRaises(ValueError):
            metrics.paired_accuracy_intervals(runs[:-1])
        with self.assertRaises(ValueError):
            metrics.paired_accuracy_intervals(runs[:-1] + [runs[0]])


class PredictionTests(unittest.TestCase):
    def frame(self):
        probability = np.eye(5) * 0.8 + 0.04
        frame = pd.DataFrame(
            {
                "graph_id": [str(k) for k in range(5)],
                "true_label": range(5),
                "predicted_label": range(5),
                "probability_argmax": range(5),
            }
        )
        for k in range(5):
            frame[f"probability_{k}"] = probability[:, k]
            frame[f"decision_{k}"] = np.log(probability[:, k])
        return frame

    def test_svm_hard_labels_are_not_replaced_by_probability_argmax(self):
        frame = self.frame()
        frame.loc[0, "predicted_label"] = 1
        aggregate, _, confusion, _ = metrics.classification_metrics(frame)
        self.assertAlmostEqual(aggregate["accuracy"], 0.8)
        self.assertEqual(aggregate["probability_argmax_disagreements"], 1)
        self.assertEqual(confusion[0, 1], 1)
        self.assertAlmostEqual(aggregate["probability_log_loss"], -math.log(0.84))

    def test_duplicate_ids_and_invalid_probabilities_rejected(self):
        frame = self.frame()
        frame.loc[1, "graph_id"] = "0"
        with self.assertRaises(ValueError):
            metrics.classification_metrics(frame)
        frame = self.frame()
        frame.loc[1, "probability_1"] = 1.1
        with self.assertRaises(ValueError):
            metrics.classification_metrics(frame)

    def test_ablation_csv_accepts_logits_without_changing_hard_predictions(self):
        frame = self.frame().drop(columns="probability_argmax")
        frame = frame.rename(columns={f"decision_{k}": f"logit_{k}" for k in range(5)})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            frame.to_csv(path, index=False)
            loaded = prediction_frame(path)
            self.assertEqual(loaded.graph_id.tolist(), ["0", "1", "2", "3", "4"])
            self.assertTrue(
                np.array_equal(loaded.predicted_label, frame.predicted_label)
            )
            self.assertEqual(metrics.classification_metrics(loaded)[0]["accuracy"], 1.0)

    def test_completion_checksum_detects_modified_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "metrics.json"
            source.write_text("{}")
            (root / "complete.json").write_text(
                json.dumps(
                    {
                        "purpose": "research",
                        "files": {
                            "metrics.json": hashlib.sha256(
                                source.read_bytes()
                            ).hexdigest()
                        },
                    }
                )
            )
            self.assertEqual(len(verified_sources(root, [source])), 2)
            source.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verified_sources(root, [source])

    def test_prepared_inputs_are_verified_before_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {"dataset_descriptors.csv": "graph_id,label\n0,0\n"}
            for seed in range(42, 47):
                for name in ("split.json", "selection.json"):
                    files[f"seed_{seed}/{name}"] = "{}"
            for name, value in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            marker = {
                "purpose": "research",
                "files": {
                    name: hashlib.sha256(value.encode()).hexdigest()
                    for name, value in files.items()
                },
            }
            (root / "complete.json").write_text(json.dumps(marker))
            verify_prepared(root)
            for name in (
                "dataset_descriptors.csv",
                "seed_42/split.json",
                "seed_43/selection.json",
            ):
                (root / name).write_text("changed")
                with self.assertRaises(ValueError):
                    verify_prepared(root)
                (root / name).write_text(files[name])

    def test_ablation_table_rejects_incomplete_results(self):
        from build_publication import ablation_table

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                ablation_table(pd.DataFrame(), Path(temporary))


if __name__ == "__main__":
    unittest.main()
