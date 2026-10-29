import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def main_run_path(results, baselines, model, seed):
    if model in ("Descriptor_SVM_tuned", "Graph2Vec_SVM"):
        return baselines / "runs" / f"seed_{seed}" / model
    return results / "runs/main" / f"seed_{seed}" / model


def prediction_frame(path):
    frame = pd.read_csv(path, dtype={"graph_id": str})
    if "probability_argmax" not in frame:
        frame["probability_argmax"] = (
            frame[[f"probability_{k}" for k in range(5)]].to_numpy().argmax(axis=1)
        )
    for k in range(5):
        if f"decision_{k}" not in frame:
            source = f"logit_{k}" if f"logit_{k}" in frame else f"probability_{k}"
            frame[f"decision_{k}"] = frame[source]
    return frame


def validate_test_identity(frame, prepared, seed):
    split = read(prepared / f"seed_{seed}" / "split.json")
    groups = split["ids"]
    sets = {key: set(map(str, groups[key])) for key in ("train", "validation", "test")}
    if split["seed"] != seed or any(
        (
            len(groups[key]) != count or len(sets[key]) != count
            for key, count in (("train", 1400), ("validation", 300), ("test", 300))
        )
    ):
        raise ValueError(f"Unexpected split sizes or seed: {seed}")
    if len(set.union(*sets.values())) != 2000:
        raise ValueError(f"Overlapping split IDs: {seed}")
    if (
        len(frame) != 300
        or frame.graph_id.duplicated().any()
        or set(frame.graph_id) != sets["test"]
    ):
        raise ValueError(f"Predictions do not match the test IDs: {seed}")
    descriptors = pd.read_csv(
        prepared / "dataset_descriptors.csv", dtype={"graph_id": str}
    ).set_index("graph_id")
    if (
        len(descriptors) != 2000
        or descriptors.index.duplicated().any()
        or set(descriptors.index) != set.union(*sets.values())
    ):
        raise ValueError("Descriptor corpus does not match the split IDs")
    truth = descriptors.loc[frame.graph_id, "label"].to_numpy()
    if not np.array_equal(frame.true_label.to_numpy(), truth):
        raise ValueError(f"Prediction labels disagree with the corpus: {seed}")
    if frame.true_label.value_counts().to_dict() != {k: 60 for k in range(5)}:
        raise ValueError("Each test set must contain 60 graphs per family")


def validate_saved_test(saved, aggregate, confusion, path):
    aliases = {
        "accuracy": "accuracy",
        "macro_precision": "macro_precision",
        "macro_recall": "macro_recall",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
    }
    for key, computed in (
        ("macro_roc_auc", "roc_auc_macro_ovr"),
        ("macro_average_precision", "average_precision_macro_ovr"),
    ):
        if key in saved:
            aliases[key] = computed
    for key, computed in aliases.items():
        if key not in saved or not np.isclose(
            saved[key], aggregate[computed], atol=1e-10, rtol=1e-08
        ):
            raise ValueError(f"Saved metric disagrees with predictions: {path} {key}")
    if not np.array_equal(saved.get("confusion_matrix"), confusion):
        raise ValueError(f"Saved confusion matrix disagrees: {path}")


def verified_sources(run, required):
    marker = read(run / "complete.json")
    if marker.get("purpose") != "research":
        raise ValueError(f"Expected a completed research run: {run}")
    inventory = marker.get("files", {})
    for path in required:
        key = str(path.relative_to(run))
        if (
            key not in inventory
            or hashlib.sha256(path.read_bytes()).hexdigest() != inventory[key]
        ):
            raise ValueError(f"Completed source checksum mismatch: {path}")
    return [*required, run / "complete.json"]


def verify_prepared(prepared):
    required = [prepared / "dataset_descriptors.csv"]
    required.extend(
        prepared / f"seed_{seed}" / name
        for seed in range(42, 47)
        for name in ("split.json", "selection.json")
    )
    verified_sources(prepared, required)
