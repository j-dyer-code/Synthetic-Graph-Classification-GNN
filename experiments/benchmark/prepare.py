import argparse
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import time
import numpy as np
from lib.common import (
    atomic_json,
    atomic_npz,
    check_original_source,
    digest,
    exclusive_lock,
    file_hash,
    immutable_json,
    load_ablation,
    load_protocol,
    source_identity,
    stable_seed,
    validate_dataset,
    verify_complete,
)

RF_PARAMETERS = dict(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
)


def iterative_rf_selection(
    X_train, y_train, names, seed, level, settings, forest_factory=None
):
    if forest_factory is None:
        from sklearn.ensemble import RandomForestClassifier

        forest_factory = RandomForestClassifier
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    if (
        X_train.ndim != 2
        or X_train.shape[1] != len(names)
        or len(X_train) != len(y_train)
        or (not len(X_train))
        or (not len(names))
        or (len(set(names)) != len(names))
        or (not np.isfinite(X_train).all())
    ):
        raise ValueError(
            "Selection requires finite training rows and unique feature names"
        )
    threshold = float(settings["threshold"])
    forests = int(settings["forests"])
    if not 0 < threshold <= 1 or forests != 5:
        raise ValueError(
            "Selection requires an inclusive threshold in (0,1] and five forests"
        )
    parameters = dict(RF_PARAMETERS)
    if settings.get("forest_parameters", parameters) != parameters:
        raise ValueError(
            "RF parameters must match the declared original-equivalent defaults"
        )
    forest_seeds = [stable_seed(seed, "rf", level, j) for j in range(forests)]
    retained = list(range(len(names)))
    passes = []
    while True:
        before = list(retained)
        importances = []
        for rf_seed in forest_seeds:
            forest = forest_factory(random_state=rf_seed, **parameters)
            forest.fit(X_train[:, before], y_train)
            values = np.asarray(forest.feature_importances_, dtype=np.float64)
            if (
                values.shape != (len(before),)
                or not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise ValueError("RF returned invalid normalized feature importances")
            total = float(values.sum())
            if total and (not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12)):
                raise ValueError("RF feature importances are not normalized")
            importances.append(values)
        averaged = np.mean(importances, axis=0)
        order = sorted(range(len(before)), key=lambda j: (-averaged[j], before[j]))
        cumulative = np.cumsum(averaged[order])
        crossings = np.flatnonzero(cumulative >= threshold)
        keep = int(crossings[0]) + 1 if len(crossings) else len(order)
        retained = [before[j] for j in order[:keep]]
        passes.append(
            dict(
                pass_number=len(passes) + 1,
                input_indices=before,
                input_names=[names[j] for j in before],
                forest_seeds=forest_seeds,
                forest_importances=[x.tolist() for x in importances],
                mean_importances=averaged.tolist(),
                ranked_indices=[before[j] for j in order],
                ranked_names=[names[before[j]] for j in order],
                cumulative_importance=cumulative.tolist(),
                selected_indices=retained,
                selected_names=[names[j] for j in retained],
                threshold=threshold,
                threshold_reached=bool(len(crossings)),
                fit_training_graphs=len(X_train),
            )
        )
        if len(retained) == len(before):
            break
    return dict(
        input_names=list(names),
        selected_indices=retained,
        selected_names=[names[j] for j in retained],
        forest_seeds=forest_seeds,
        rf_parameters=parameters,
        threshold=threshold,
        passes=passes,
        fit_scope="Training graphs only",
        fit_training_graphs=len(X_train),
        selection_rule="Inclusive first cumulative importance >= threshold; repeat until feature count is stable",
        tie_rule="Original candidate index ascending",
    )


def collect_graph_features(data):
    node_means, graph_values = ([], [])
    for index in range(len(data.graphs)):
        raw = data.raw(index)
        node_means.append(np.asarray(raw["x_topology"], dtype=np.float64).mean(axis=0))
        graph_values.append(np.asarray(raw["graph_features"], dtype=np.float64))
    node_means, graph_values = (np.asarray(node_means), np.asarray(graph_values))
    if (
        node_means.shape != (len(data.graphs), 5)
        or graph_values.shape != node_means.shape
        or (not np.isfinite(node_means).all())
        or (not np.isfinite(graph_values).all())
    ):
        raise ValueError(
            "Expected five finite raw node means and five graph values per graph"
        )
    return (node_means, graph_values)


def save_descriptors(path, data, node_means, graph_values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["graph_id", "label"]
            + [f"node_mean_{i}" for i in range(5)]
            + [f"graph_{i}" for i in range(5)]
        )
        for i, graph in enumerate(data.graphs):
            writer.writerow(
                [str(graph["id"]), int(data.labels[i])]
                + node_means[i].tolist()
                + graph_values[i].tolist()
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_seed(data, split, seed, protocol, ablation, node_means, graph_values):
    train = np.asarray(split["train"], dtype=np.int64)
    settings = protocol["feature_selection"]
    node_selection = iterative_rf_selection(
        node_means[train], data.labels[train], data.node_names, seed, "node", settings
    )
    graph_selection = iterative_rf_selection(
        graph_values[train],
        data.labels[train],
        data.graph_names,
        seed,
        "graph",
        settings,
    )
    node_stats = ablation.fit_scaler((data.raw(int(i))["x_topology"] for i in train), 5)
    graph_stats = ablation.fit_scaler((graph_values[train],), 5)
    raw_svm = np.concatenate((node_means, graph_values), axis=1)
    all_svm_stats = ablation.fit_scaler((raw_svm[train],), 10)
    columns = list(node_selection["selected_indices"]) + [
        5 + i for i in graph_selection["selected_indices"]
    ]
    full_names = ["node_mean:" + name for name in data.node_names] + [
        "graph:" + name for name in data.graph_names
    ]
    svm_stats = {
        key: [values[i] for i in columns] if isinstance(values, list) else values
        for key, values in all_svm_stats.items()
    }
    X_all = (raw_svm - np.asarray(all_svm_stats["mean"])) / np.asarray(
        all_svm_stats["scale"]
    )
    X = X_all[:, columns]
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite normalized SVM arrays")
    training_ids = [str(data.graphs[int(i)]["id"]) for i in train]
    selection = dict(
        seed=seed,
        node=node_selection,
        graph=graph_selection,
        training_graph_ids=training_ids,
        node_selection_input="Per-graph means of all five raw node candidates; one row per training graph",
        graph_selection_input="All five raw graph candidates; one row per training graph",
        svm_candidate_names=full_names,
        svm_selected_indices=columns,
        svm_selected_names=[full_names[i] for i in columns],
        test_labels_used=False,
        validation_labels_used=False,
    )
    normalisation = dict(
        node=node_stats,
        graph=graph_stats,
        svm=svm_stats,
        svm_all_candidates=all_svm_stats,
        svm_selected_indices=columns,
        svm_selected_names=[full_names[i] for i in columns],
        fit_training_graph_ids=training_ids,
        node_fit_scope="Every node of training graphs only; equal weight per node",
        graph_fit_scope="Training graphs only; equal weight per graph",
        svm_fit_scope="Training rows of ten raw per-graph candidates; selected columns retained after scaling",
        variance_definition="Population variance; scale <= 1e-12 replaced by 1",
    )
    return (
        selection,
        normalisation,
        dict(
            graph_ids=np.asarray([str(g["id"]) for g in data.graphs], dtype=str),
            labels=np.asarray(data.labels, dtype=np.int64),
            X=np.asarray(X, dtype=np.float64),
        ),
    )


def split_record(data, indices, seed, fingerprint, protocol):
    return dict(
        seed=seed,
        dataset_fingerprint=fingerprint,
        requested_fractions=dict(
            train=protocol["data"]["train_fraction"],
            validation=protocol["data"]["validation_fraction"],
            test=1
            - protocol["data"]["train_fraction"]
            - protocol["data"]["validation_fraction"],
        ),
        ids={
            part: [str(data.graphs[i]["id"]) for i in rows]
            for part, rows in indices.items()
        },
        class_counts={
            part: np.bincount(
                data.labels[rows], minlength=len(data.class_names)
            ).tolist()
            for part, rows in indices.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stratified partitions, training-only feature selection and standardisation."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--new-corpus",
        action="store_true",
        help="Accept a new 400-per-family corpus instead of the published graph identities",
    )
    parser.add_argument(
        "--software-test",
        "--smoke",
        dest="smoke",
        action="store_true",
        help="Prepare the explicit 50-graph software fixture; never a research completion",
    )
    args = parser.parse_args()
    protocol = load_protocol(smoke=args.smoke)
    source_hashes = check_original_source()
    ablation = load_ablation()
    path = args.dataset.expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    data = ablation.PreparedDataset(path, cache_graphs=0)
    print("Validating every prepared graph and its checksum...", flush=True)
    audit = data.validate()
    validate_dataset(data, audit, protocol, new_corpus=args.new_corpus)
    identity = dict(
        schema_version=1,
        purpose=protocol["purpose"],
        protocol=digest(protocol),
        dataset_scope="software_test"
        if args.smoke
        else "new_corpus"
        if args.new_corpus
        else "published",
        original_source=source_hashes,
        preparation_source=source_identity(),
        preparation_environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scikit_learn=importlib.metadata.version("scikit-learn"),
            scipy=importlib.metadata.version("scipy"),
        ),
        audited_utility_sha256=file_hash(Path(ablation.__file__)),
        dataset_fingerprint=audit["fingerprint"],
        dataset_identity=audit["identity"],
    )
    fingerprint = digest(identity)
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.output / ".prepare.lock"):
        if (args.output / "complete.json").exists():
            verify_complete(args.output, fingerprint)
            print(f"Already complete and verified: {args.output}", flush=True)
            return
        immutable_json(args.output / "identity.json", identity)
        immutable_json(args.output / "protocol.json", protocol)
        immutable_json(args.output / "dataset_audit.json", audit)
        node_means, graph_values = collect_graph_features(data)
        save_descriptors(
            args.output / "dataset_descriptors.csv", data, node_means, graph_values
        )
        scientific = [
            "identity.json",
            "protocol.json",
            "dataset_audit.json",
            "dataset_descriptors.csv",
        ]
        for seed in protocol["seeds"]:
            started = time.perf_counter()
            indices = ablation.stratified_split(
                data,
                seed,
                protocol["data"]["train_fraction"],
                protocol["data"]["validation_fraction"],
            )
            target = args.output / f"seed_{seed}"
            target.mkdir(exist_ok=True)
            immutable_json(
                target / "split.json",
                split_record(data, indices, seed, audit["fingerprint"], protocol),
            )
            selection, normalisation, arrays = prepare_seed(
                data, indices, seed, protocol, ablation, node_means, graph_values
            )
            immutable_json(target / "selection.json", selection)
            immutable_json(target / "normalisation.json", normalisation)
            atomic_npz(target / "svm_arrays.npz", **arrays)
            scientific.extend(
                (
                    f"seed_{seed}/{name}"
                    for name in (
                        "split.json",
                        "selection.json",
                        "normalisation.json",
                        "svm_arrays.npz",
                    )
                )
            )
            print(
                f"Prepared seed {seed}: {len(selection['node']['selected_indices'])} node and {len(selection['graph']['selected_indices'])} graph features ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        atomic_json(
            args.output / "complete.json",
            dict(
                schema_version=1,
                purpose=protocol["purpose"],
                fingerprint=fingerprint,
                files={name: file_hash(args.output / name) for name in scientific},
            ),
        )
        verify_complete(args.output, fingerprint)
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
