import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import classifiers


def fixture():
    rng = np.random.default_rng(983)
    labels = np.repeat(np.arange(5), 12)
    features = rng.normal(scale=0.7, size=(60, 6))
    features[:, :5] += 2 * np.eye(5)[labels]
    ids = [str(i) for i in range(60)]
    split = {name: [] for name in ("train", "validation", "test")}
    for k in range(5):
        split["train"].extend(ids[12 * k : 12 * k + 8])
        split["validation"].extend(ids[12 * k + 8 : 12 * k + 10])
        split["test"].extend(ids[12 * k + 10 : 12 * k + 12])
    training = np.asarray([int(gid) for gid in split["train"]])
    mean, scale = (features[training].mean(axis=0), features[training].std(axis=0))
    features = (features - mean) / np.where(scale == 0, 1.0, scale)
    return (features, labels, ids, split)


class SelectionIsolationTests(unittest.TestCase):
    def test_changed_test_features_and_labels_do_not_change_selection(self):
        features, labels, ids, split = fixture()
        changed_features, changed_labels = (features.copy(), labels.copy())
        test = np.asarray([int(gid) for gid in split["test"]])
        changed_features[test] = np.random.default_rng(981).normal(
            50.0, 3.0, (len(test), features.shape[1])
        )
        changed_labels[test] = (changed_labels[test] + 1) % 5
        with tempfile.TemporaryDirectory(prefix="svm-selection-fixture-") as temporary:
            folder = Path(temporary)
            for name, x, y in (
                ("a", features, labels),
                ("b", changed_features, changed_labels),
            ):
                classifiers.fit_classifier(
                    x, y, ids, split, 42, folder / name, "Graph2Vec_SVM", "a" * 64
                )
            grids = [
                json.loads((folder / name / "grid.json").read_text())
                for name in ("a", "b")
            ]
            parameters = [
                json.loads((folder / name / "parameters.json").read_text())
                for name in ("a", "b")
            ]
            for grid in grids:
                self.assertEqual(len(grid["candidates"]), 25)
                self.assertFalse(grid["test_used_for_selection"])

            def selection_record(grid):
                return [
                    {k: v for k, v in row.items() if k != "fit_and_validation_seconds"}
                    for row in grid["candidates"]
                ]

            self.assertEqual(selection_record(grids[0]), selection_record(grids[1]))
            self.assertEqual(
                grids[0]["selected_candidate"], grids[1]["selected_candidate"]
            )
            self.assertEqual(parameters[0], parameters[1])
            predictions = [
                pd.read_csv(folder / name / "test_predictions.csv")
                for name in ("a", "b")
            ]
            self.assertFalse(
                np.array_equal(predictions[0].true_label, predictions[1].true_label)
            )

    def test_actual_svc_validation_ties_choose_first_grid_setting(self):
        features, labels, ids, split = fixture()
        features[:] = 0.0
        with tempfile.TemporaryDirectory(prefix="svm-tie-fixture-") as temporary:
            folder = Path(temporary)
            classifiers.fit_classifier(
                features,
                labels,
                ids,
                split,
                43,
                folder,
                "Descriptor_SVM_tuned",
                "b" * 64,
            )
            grid = json.loads((folder / "grid.json").read_text())
            parameters = json.loads((folder / "parameters.json").read_text())
            self.assertEqual(
                len({row["validation_accuracy"] for row in grid["candidates"]}), 1
            )
            self.assertEqual(grid["selected_candidate"], 0)
            self.assertEqual(parameters["selected_candidate"], 0)
            self.assertEqual(parameters["C"], classifiers.C_VALUES[0])
            self.assertEqual(
                parameters["gamma_multiplier"], classifiers.GAMMA_MULTIPLIERS[0]
            )
            self.assertEqual(parameters["gamma_base"], 1.0)


class LabelValidationTests(unittest.TestCase):
    def test_fractional_training_and_validation_labels_are_rejected(self):
        for partition in ("train", "validation"):
            with self.subTest(
                partition=partition
            ), tempfile.TemporaryDirectory() as directory:
                features, labels, ids, split = fixture()
                labels = labels.astype(float)
                labels[int(split[partition][0])] = 0.5
                output = Path(directory)
                with self.assertRaisesRegex(ValueError, "finite integers"):
                    classifiers.fit_classifier(
                        features,
                        labels,
                        ids,
                        split,
                        42,
                        output,
                        "Graph2Vec_SVM",
                        "a" * 64,
                    )
                self.assertFalse((output / "grid.json").exists())

    def test_held_out_label_validation_occurs_after_model_selection(self):
        features, labels, ids, split = fixture()
        labels = labels.astype(float)
        labels[int(split["test"][0])] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "finite integers"):
                classifiers.fit_classifier(
                    features, labels, ids, split, 42, output, "Graph2Vec_SVM", "a" * 64
                )
            self.assertTrue((output / "grid.json").exists())
            self.assertFalse((output / "test_predictions.csv").exists())

    def test_metrics_reject_fractional_and_nonfinite_class_labels(self):
        labels = np.arange(5, dtype=float)
        probabilities = np.eye(5)
        for bad_value in (0.5, np.nan, np.inf):
            changed = labels.copy()
            changed[0] = bad_value
            for actual, predicted in ((changed, labels), (labels, changed)):
                with self.subTest(value=bad_value), self.assertRaisesRegex(
                    ValueError, "finite integers"
                ):
                    classifiers.classification_metrics(
                        actual, predicted, probabilities, classifiers.CLASSES
                    )


if __name__ == "__main__":
    unittest.main()
