import argparse
import gzip
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from common import (
    atomic_bytes,
    atomic_json,
    atomic_npz,
    canonical,
    complete,
    digest,
    environment,
    exclusive_lock,
    file_hash,
    immutable_json,
    load_protocol,
    source_identity,
    stable_seed,
    verify_complete,
)


def token_path(output, graph_id):
    return output / "tokens" / (digest(str(graph_id)) + ".json.gz")


def read_tokens(output, graph_id, inventory):
    path = token_path(output, graph_id)
    expected = inventory[str(graph_id)]["tokens_sha256"]
    if file_hash(path) != expected:
        raise ValueError(f"Changed topology tokens for graph {graph_id}")
    return json.loads(gzip.decompress(path.read_bytes()))


def dataset_manifest(dataset):
    path = Path(dataset).expanduser().resolve()
    path = path / "manifest.json" if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(
            "Pass --dataset as the graph directory or manifest.json"
        )
    return path


def validate_prepared(prepared, protocol):
    marker = verify_complete(prepared)
    if marker["purpose"] != protocol["purpose"]:
        raise ValueError("Preparation purpose differs from experiment purpose")
    identity = json.loads((prepared / "identity.json").read_text())
    saved_protocol = json.loads((prepared / "protocol.json").read_text())
    required = {
        "identity.json",
        "protocol.json",
        "dataset_audit.json",
        "dataset_descriptors.csv",
    }
    required.update(
        (
            f"seed_{seed}/{name}"
            for seed in protocol["seeds"]
            for name in (
                "split.json",
                "selection.json",
                "normalisation.json",
                "svm_arrays.npz",
            )
        )
    )
    if set(marker["files"]) != required or digest(identity) != marker["fingerprint"]:
        raise ValueError("Preparation completion inventory or identity mismatch")
    if (
        identity.get("purpose") != protocol["purpose"]
        or saved_protocol.get("classes") != protocol["classes"]
        or saved_protocol.get("seeds") != protocol["seeds"]
    ):
        raise ValueError("Preparation purpose, classes or seeds differ")
    return marker


def load_inputs(prepared, output, smoke=False):
    protocol = load_protocol(smoke)
    marker = validate_prepared(prepared, protocol)
    identity = json.loads((output / "identity.json").read_text())
    current = dict(
        protocol=protocol,
        source=source_identity(),
        environment=environment(),
        prepared_fingerprint=marker["fingerprint"],
        prepared_complete_sha256=file_hash(prepared / "complete.json"),
    )
    for key, value in current.items():
        if identity[key] != value:
            raise ValueError(
                f"Changed experiment {key}; resume requires the same inputs, source and environment"
            )
    return (protocol, prepared, identity)


def prepare(prepared, output, dataset, smoke=False):
    from representations import wl_tokens

    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".prepare.lock"):
        protocol = load_protocol(smoke)
        previous = validate_prepared(prepared, protocol)
        manifest_path = dataset_manifest(dataset)
        manifest = json.loads(manifest_path.read_text())
        graphs = manifest["graphs"]
        if (
            manifest.get("status") != "complete"
            or manifest["class_names"] != protocol["classes"]
        ):
            raise ValueError("Invalid raw manifest status or class order")
        if smoke and manifest.get("purpose") != "software_test":
            raise ValueError("Fixture mode requires explicitly labelled software data")
        ids = [str(g["id"]) for g in graphs]
        labels = [g["label"] for g in graphs]
        if (
            len(ids) != len(set(ids))
            or np.bincount(labels, minlength=5).tolist()
            != [protocol["data"]["graphs_per_class"]] * 5
        ):
            raise ValueError("Unexpected graph identifiers or class counts")
        source_files = []
        for graph in graphs:
            path = (manifest_path.parent / graph["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent) or not path.is_file():
                raise ValueError("Invalid graph file path")
            actual = file_hash(path)
            if actual != graph["sha256"]:
                raise ValueError(f"Raw graph checksum differs: {graph['id']}")
            source_files.append(
                dict(id=str(graph["id"]), label=graph["label"], sha256=actual)
            )
        raw_identity = dict(
            manifest_sha256=file_hash(manifest_path), graph_files=source_files
        )
        fingerprint = digest(raw_identity)
        audit = json.loads((prepared / "dataset_audit.json").read_text())
        if raw_identity != audit["identity"]:
            raise ValueError("Raw graphs do not match the shared benchmark preparation")
        by_id = dict(zip(ids, labels))
        all_ids = set(ids)
        for seed in protocol["seeds"]:
            split = json.loads((prepared / f"seed_{seed}/split.json").read_text())[
                "ids"
            ]
            flattened = [
                gid for part in ("train", "validation", "test") for gid in split[part]
            ]
            if len(flattened) != len(set(flattened)) or set(flattened) != all_ids:
                raise ValueError(
                    "Split overlap, missing graphs or different identifiers"
                )
            if {k: len(split[k]) for k in split} != protocol["data"]["split_sizes"]:
                raise ValueError("Unexpected saved split sizes")
            with np.load(
                prepared / f"seed_{seed}/svm_arrays.npz", allow_pickle=False
            ) as arrays:
                saved_ids = arrays["graph_ids"].astype(str).tolist()
                saved_labels = arrays["labels"]
                if (
                    set(saved_ids) != all_ids
                    or len(saved_ids) != len(all_ids)
                    or any(
                        (by_id[g] != int(y) for g, y in zip(saved_ids, saved_labels))
                    )
                ):
                    raise ValueError(
                        "Descriptor arrays differ from raw corpus identities"
                    )
                if arrays["X"].ndim != 2 or not np.isfinite(arrays["X"]).all():
                    raise ValueError("Invalid saved descriptor matrix")
        identity = dict(
            purpose=protocol["purpose"],
            protocol=protocol,
            source=source_identity(),
            environment=environment(),
            prepared_fingerprint=previous["fingerprint"],
            prepared_complete_sha256=file_hash(prepared / "complete.json"),
            raw_identity=raw_identity,
            dataset_fingerprint=fingerprint,
        )
        immutable_json(output / "identity.json", identity)
        if (output / "prepared/complete.json").exists():
            verify_complete(output / "prepared", digest(identity))
            print("PREPARE VERIFIED: cached topology tokens", flush=True)
            return
        inventory = {}
        start = time.perf_counter()
        for index, graph in enumerate(graphs):
            gid = str(graph["id"])
            path = token_path(output, gid)
            meta = path.with_suffix(".meta.json")
            item_identity = dict(
                graph_sha256=graph["sha256"],
                wl_iterations=protocol["embedding"]["wl_iterations"],
                experiment=digest(identity),
            )
            if path.exists() and meta.exists():
                item = json.loads(meta.read_text())
                if item["identity"] != item_identity or item[
                    "tokens_sha256"
                ] != file_hash(path):
                    raise ValueError(f"Invalid partial token checkpoint for {gid}")
            else:
                with np.load(
                    manifest_path.parent / graph["path"], allow_pickle=False
                ) as arrays:
                    edges = arrays["edge_index"]
                    n = len(arrays["x_topology"])
                if (
                    edges.ndim != 2
                    or edges.shape[0] != 2
                    or edges.dtype.kind not in "iu"
                    or (n < 2)
                ):
                    raise ValueError("Invalid adjacency shape")
                if edges.size and (
                    edges.min() < 0 or edges.max() >= n or (edges[0] == edges[1]).any()
                ):
                    raise ValueError("Invalid edge endpoints or self loops")
                codes = edges[0].astype(np.int64) * n + edges[1]
                if len(np.unique(codes)) != len(codes) or not np.array_equal(
                    np.sort(codes), np.sort(edges[1] * n + edges[0])
                ):
                    raise ValueError(
                        "Adjacency must contain each direction exactly once"
                    )
                tokens = wl_tokens(
                    edges, n, iterations=protocol["embedding"]["wl_iterations"]
                )
                if len(tokens) != n * (protocol["embedding"]["wl_iterations"] + 1):
                    raise ValueError("WL token multiplicity lost")
                atomic_bytes(path, gzip.compress(canonical(tokens).encode(), mtime=0))
                item = dict(
                    identity=item_identity,
                    tokens_sha256=file_hash(path),
                    nodes=n,
                    token_count=len(tokens),
                )
                atomic_json(meta, item)
            inventory[gid] = dict(item, token_file=str(path.relative_to(output)))
            if (index + 1) % 100 == 0 or index + 1 == len(graphs):
                print(
                    f"TOKENS {index + 1}/{len(graphs)} elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        folder = output / "prepared"
        folder.mkdir(exist_ok=True)
        atomic_json(folder / "token_inventory.json", inventory)
        atomic_json(
            folder / "corpus.json",
            dict(graph_ids=ids, labels=labels, classes=protocol["classes"]),
        )
        atomic_json(
            folder / "timings.json",
            dict(token_preparation_this_attempt_seconds=time.perf_counter() - start),
        )
        complete(
            folder,
            digest(identity),
            protocol["purpose"],
            ["token_inventory.json", "corpus.json", "timings.json"],
        )
        print(f"PREPARE COMPLETE: {len(graphs)} topology documents", flush=True)


class Documents:
    def __init__(self, output, ids, inventory):
        self.output = output
        self.ids = ids
        self.inventory = inventory
        self.cache = {}

    def __iter__(self):
        for gid in self.ids:
            if gid not in self.cache:
                self.cache[gid] = read_tokens(self.output, gid, self.inventory)
            yield (gid, self.cache[gid])


def run_seed(prepared, output, seed, smoke=False):
    from representations import (
        train_model,
        infer_graph,
        vocabulary_coverage,
        frozen_weights_hash,
    )
    from classifiers import fit_classifier

    protocol, prepared, identity = load_inputs(prepared, output, smoke)
    if seed not in protocol["seeds"]:
        raise ValueError("Seed is outside the declared protocol")
    verify_complete(output / "prepared", digest(identity))
    inventory = json.loads((output / "prepared/token_inventory.json").read_text())
    corpus = json.loads((output / "prepared/corpus.json").read_text())
    ids = corpus["graph_ids"]
    labels = np.array(corpus["labels"])
    lookup = {g: i for i, g in enumerate(ids)}
    split = json.loads((prepared / f"seed_{seed}/split.json").read_text())["ids"]
    root = output / f"runs/seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".seed.lock"):
        seed_config = dict(
            purpose=protocol["purpose"],
            seed=seed,
            experiment_fingerprint=digest(identity),
            split_ids=split,
        )
        immutable_json(root / "seed_config.json", seed_config)
        representation = root / "representation"
        representation.mkdir(exist_ok=True)
        if (representation / "complete.json").exists():
            verify_complete(representation, digest(seed_config))
            with np.load(representation / "embeddings.npz", allow_pickle=False) as data:
                if data["graph_ids"].astype(str).tolist() != ids:
                    raise ValueError("Embedding graph order differs")
                X = data["embeddings"].copy()
            print(f"EMBEDDING VERIFIED seed={seed}", flush=True)
        else:
            config = dict(
                protocol["embedding"], checkpoint_identity=digest(seed_config)
            )
            documents = Documents(output, split["train"], inventory)
            start = time.perf_counter()
            model = train_model(
                documents,
                stable_seed(seed, "graph2vec", "train"),
                config,
                root / "model_checkpoints",
            )
            train_seconds = time.perf_counter() - start
            frozen = frozen_weights_hash(model)
            X = np.empty((len(ids), config["dimension"]), dtype=np.float32)
            coverage = {}
            infer_times = []
            cache = root / "inferred"
            cache.mkdir(exist_ok=True)
            for index, gid in enumerate(ids):
                vector_path = cache / (digest(gid) + ".npz")
                meta = cache / (digest(gid) + ".json")
                vector_identity = digest(
                    dict(seed=seed_config, graph=gid, weights=frozen)
                )
                if vector_path.exists() and meta.exists():
                    saved = json.loads(meta.read_text())
                    if saved["identity"] != vector_identity or saved[
                        "sha256"
                    ] != file_hash(vector_path):
                        raise ValueError("Invalid inferred vector checkpoint")
                    with np.load(vector_path, allow_pickle=False) as data:
                        vector = data["vector"].copy()
                else:
                    tokens = documents.cache.get(gid)
                    if tokens is None:
                        tokens = read_tokens(output, gid, inventory)
                    start = time.perf_counter()
                    vector = infer_graph(
                        model,
                        tokens,
                        stable_seed(seed, "graph2vec", "infer", gid),
                        config,
                    )
                    saved = dict(
                        identity=vector_identity,
                        coverage=vocabulary_coverage(model, tokens),
                        seconds=time.perf_counter() - start,
                    )
                    atomic_npz(vector_path, vector=vector)
                    saved["sha256"] = file_hash(vector_path)
                    atomic_json(meta, saved)
                if (
                    vector.shape != (config["dimension"],)
                    or not np.isfinite(vector).all()
                ):
                    raise ValueError("Invalid inferred vector")
                X[index] = vector
                coverage[gid] = saved["coverage"]
                infer_times.append(saved["seconds"])
                if (index + 1) % 100 == 0 or index + 1 == len(ids):
                    print(f"INFER seed={seed} {index + 1}/{len(ids)}", flush=True)
            if frozen_weights_hash(model) != frozen:
                raise ValueError("Representation weights changed during inference")
            atomic_npz(
                representation / "embeddings.npz",
                graph_ids=np.array(ids),
                labels=labels,
                embeddings=X,
            )
            atomic_json(
                representation / "coverage.json",
                dict(per_graph=coverage, split_ids=split),
            )
            atomic_json(
                representation / "model_info.json",
                dict(
                    config=config,
                    seed=stable_seed(seed, "graph2vec", "train"),
                    frozen_weights_sha256=frozen,
                    vocabulary_size=len(model.wv),
                    fit_graph_ids=split["train"],
                    inference_graph_ids=ids,
                    weights_frozen_verified=True,
                    training_checkpoint=model.structural_svm_checkpoint,
                    training_seconds_this_attempt=train_seconds,
                    inference_seconds_per_graph=infer_times,
                    model_files={
                        str(p.relative_to(root)): file_hash(p)
                        for p in sorted((root / "model_checkpoints").glob("*.model"))
                        if p.is_file()
                    },
                ),
            )
            complete(
                representation,
                digest(seed_config),
                protocol["purpose"],
                ["embeddings.npz", "coverage.json", "model_info.json"],
            )
            del model, documents
            print(f"EMBEDDING COMPLETE seed={seed}", flush=True)
        train_index = [lookup[g] for g in split["train"]]
        scaler = StandardScaler().fit(X[train_index])
        graph_X = scaler.transform(X).astype(np.float64)
        for name in protocol["representations"]:
            folder = root / name
            folder.mkdir(exist_ok=True)
            config = dict(
                seed_config,
                representation=name,
                classes=protocol["classes"],
                protocol=protocol,
                representation_fingerprint=verify_complete(representation)[
                    "fingerprint"
                ],
                prepared_fingerprint=identity["prepared_fingerprint"],
            )
            immutable_json(folder / "config.json", config)
            if (folder / "complete.json").exists():
                verify_complete(folder, digest(config))
                print(f"SVM VERIFIED seed={seed} {name}", flush=True)
                continue
            if name == "Graph2Vec_SVM":
                classifier_X = graph_X
                atomic_json(
                    folder / "scaler.json",
                    dict(
                        fit_graph_ids=split["train"],
                        mean=scaler.mean_.tolist(),
                        scale=scaler.scale_.tolist(),
                        variance=scaler.var_.tolist(),
                    ),
                )
            else:
                with np.load(
                    prepared / f"seed_{seed}/svm_arrays.npz", allow_pickle=False
                ) as data:
                    order = {g: i for i, g in enumerate(data["graph_ids"].astype(str))}
                    classifier_X = data["X"][[order[g] for g in ids]].astype(np.float64)
                normalization = json.loads(
                    (prepared / f"seed_{seed}/normalisation.json").read_text()
                )
                atomic_json(
                    folder / "scaler.json",
                    dict(
                        source="saved training-only selected descriptor scaling",
                        normalisation=normalization,
                        source_sha256=file_hash(
                            prepared / f"seed_{seed}/normalisation.json"
                        ),
                    ),
                )
            fit_classifier(
                classifier_X, labels, ids, split, seed, folder, name, digest(config)
            )
            complete(
                folder,
                digest(config),
                protocol["purpose"],
                [
                    "config.json",
                    "scaler.json",
                    "grid.json",
                    "parameters.json",
                    "svm.joblib",
                    "metrics.json",
                    "test_predictions.csv",
                ],
            )
            print(f"SVM COMPLETE seed={seed} {name}", flush=True)
        complete(
            root,
            digest(seed_config),
            protocol["purpose"],
            [
                "seed_config.json",
                "representation/complete.json",
                "Graph2Vec_SVM/complete.json",
                "Descriptor_SVM_tuned/complete.json",
            ],
        )
        print(f"SEED COMPLETE {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and run tuned descriptor-SVM and Graph2Vec-SVM baselines."
    )
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Shared benchmark prepared directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Baseline checkpoint and result directory",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepared = args.prepared.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Launch with PYTHONHASHSEED=0")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"Launch with {key}=1")
    if args.command == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset")
        prepare(prepared, output, args.dataset, args.smoke)
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(prepared, output, args.seed, args.smoke)


if __name__ == "__main__":
    main()
