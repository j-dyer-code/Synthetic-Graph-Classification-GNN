import argparse
import fcntl
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import numpy as np
from lib.common import (
    atomic_bytes,
    atomic_json,
    classification_metrics,
    digest,
    file_hash,
    immutable_json,
    load_ablation,
    stable_seed,
    validate_dataset,
    verify_complete,
    verify_preparation,
)
from support import load_protocol, package_identity
from run_main import write_predictions


def run(args):
    import joblib
    from sklearn.svm import SVC

    protocol = load_protocol(args.software_test)["main"]
    if args.seed not in protocol["seeds"]:
        raise ValueError("Seed outside protocol")
    prepared_path = args.prepared or args.output / "prepared"
    manifest = args.dataset / "manifest.json" if args.dataset.is_dir() else args.dataset
    marker, identity, audit = verify_preparation(prepared_path, manifest, protocol)
    data = load_ablation().PreparedDataset(manifest, cache_graphs=0)
    validate_dataset(
        data, audit, protocol, new_corpus=identity["dataset_scope"] == "new_corpus"
    )
    if data.validate()["fingerprint"] != audit["fingerprint"]:
        raise ValueError("Prepared graph data changed")
    seed_dir = prepared_path / f"seed_{args.seed}"
    splits = json.loads((seed_dir / "split.json").read_text())["ids"]
    selected = json.loads((seed_dir / "selection.json").read_text())
    normalisation = json.loads((seed_dir / "normalisation.json").read_text())
    if (
        selected["training_graph_ids"] != splits["train"]
        or normalisation["fit_training_graph_ids"] != splits["train"]
    ):
        raise ValueError(
            "Selected features or scaling differ from the training partition"
        )
    with np.load(seed_dir / "svm_arrays.npz", allow_pickle=False) as arrays:
        X, labels = (arrays["X"], arrays["labels"])
        ids = arrays["graph_ids"].tolist()
    if ids != [str(g["id"]) for g in data.graphs] or not np.array_equal(
        labels, data.labels
    ):
        raise ValueError("SVM row mapping differs from the dataset")
    lookup = {v: i for i, v in enumerate(ids)}
    split = {part: [lookup[v] for v in rows] for part, rows in splits.items()}
    if sorted(sum(split.values(), [])) != list(range(len(ids))):
        raise ValueError("Partitions must be disjoint and exhaustive")
    seed = stable_seed(args.seed, "svm")
    config = dict(
        schema_version=1,
        purpose=protocol["purpose"],
        pipeline="main",
        seed=args.seed,
        architecture="SVM",
        prepared_fingerprint=marker["fingerprint"],
        identity=dict(
            protocol=digest(protocol),
            execution_source=package_identity(),
            dataset_fingerprint=audit["fingerprint"],
        ),
        split_ids=splits,
        selected_features=selected,
        normalisation=normalisation,
        seeds=dict(master=args.seed, svm=seed),
        protocol=protocol,
        device="cpu",
        python=platform.python_version(),
        software={
            p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")
        },
    )
    fingerprint = digest(config)
    folder = args.output / "runs/main" / f"seed_{args.seed}" / "SVM"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / "complete.json").exists():
            verify_complete(folder, fingerprint)
            print(f"Verified complete: {folder}", flush=True)
            return
        immutable_json(folder / "run_config.json", config)
        model = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
            decision_function_shape="ovr",
        )
        started = time.perf_counter()
        model.fit(X[split["train"]], labels[split["train"]])
        fit_seconds = time.perf_counter() - started
        if model.classes_.tolist() != list(range(5)):
            raise ValueError("SVM class order differs")
        output = {}
        inference = {}
        for part in ("validation", "test"):
            rows = split[part]
            started = time.perf_counter()
            predicted = model.predict(X[rows])
            probabilities = model.predict_proba(X[rows])
            decisions = model.decision_function(X[rows])
            inference[part] = time.perf_counter() - started
            write_predictions(
                folder / f"{part}_predictions.csv",
                splits[part],
                labels[rows],
                predicted,
                probabilities,
                decisions,
            )
            output[part] = classification_metrics(
                labels[rows], predicted, probabilities, protocol["classes"]
            )
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        atomic_bytes(folder / "svm.joblib", buffer.getvalue())
        atomic_json(
            folder / "best_parameters.json", dict(protocol["svm"], random_state=seed)
        )
        atomic_json(folder / "history.json", [])
        atomic_json(
            folder / "tuning.json",
            dict(tuned=False, trials=[], best_trial=None, tuning_seconds=0.0),
        )
        atomic_json(
            folder / "metrics.json",
            dict(
                fingerprint=fingerprint,
                classes=protocol["classes"],
                best_epoch=None,
                epochs_executed=None,
                best_validation_loss=None,
                parameter_count=None,
                timings=dict(
                    tuning_seconds=0.0,
                    svm_fit_seconds=fit_seconds,
                    test_inference_seconds=inference["test"],
                ),
                timing_scopes=dict(
                    tuning_seconds="No hyperparameter search",
                    svm_fit_seconds="SVC.fit including training-only internal probability calibration",
                    test_inference_seconds="Held-out prediction and probability/decision extraction",
                ),
                **output,
            ),
        )
        names = [
            "run_config.json",
            "best_parameters.json",
            "history.json",
            "tuning.json",
            "metrics.json",
            "test_predictions.csv",
            "validation_predictions.csv",
            "svm.joblib",
        ]
        atomic_json(
            folder / "complete.json",
            dict(
                schema_version=1,
                purpose=protocol["purpose"],
                status="complete",
                pipeline="main",
                fingerprint=fingerprint,
                files={name: file_hash(folder / name) for name in names},
            ),
        )
        verify_complete(folder, fingerprint)
    print(f"Complete: {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the fixed descriptor SVM on a prepared partition."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--software-test", action="store_true")
    run(parser.parse_args())
