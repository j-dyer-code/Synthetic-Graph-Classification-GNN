import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import socket
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import OrderedDict
import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def immutable_json(path, value):
    if Path(path).exists():
        existing = json.loads(Path(path).read_text())
        if canonical(existing) != canonical(value):
            raise ValueError(
                f"Existing configuration differs: {path}. Use another output directory."
            )
    else:
        atomic_json(path, value)


class PreparedDataset:
    def __init__(self, manifest_path, cache_graphs):
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("status") != "complete":
            raise ValueError("Prepared manifest must have status=complete")
        self.graphs = self.manifest["graphs"]
        self.class_names = self.manifest["class_names"]
        self.node_names = self.manifest["node_feature_names"]
        self.graph_names = self.manifest["graph_feature_names"]
        self.cache_size = cache_graphs
        self.cache = OrderedDict()
        if len(self.node_names) != 5 or len(self.graph_names) != 5:
            raise ValueError(
                "This protocol requires five fixed candidate node descriptors and five graph descriptors"
            )
        if len(self.class_names) != 5 or not self.graphs:
            raise ValueError(
                "This five-family protocol requires exactly five classes and nonempty graphs"
            )
        ids = [str(g["id"]) for g in self.graphs]
        if len(set(ids)) != len(ids):
            raise ValueError("Graph IDs must be unique")
        self.paths = []
        for graph in self.graphs:
            if not isinstance(graph["label"], int) or not 0 <= graph["label"] < len(
                self.class_names
            ):
                raise ValueError(f"Invalid integer class label for {graph['id']}")
            path = (self.root / graph["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(
                    f"Graph path must resolve to an existing file within the dataset: {path}"
                )
            self.paths.append(path)
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Each graph must have its own NPZ path")
        self.labels = np.asarray([g["label"] for g in self.graphs], dtype=np.int64)
        if set(self.labels.tolist()) != set(range(len(self.class_names))):
            raise ValueError("Every declared class must occur in the manifest")

    def raw(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        with np.load(self.paths[index], allow_pickle=False) as archive:
            item = {
                key: np.asarray(archive[key])
                for key in ("edge_index", "x_topology", "graph_features", "community")
            }
        if self.cache_size != 0:
            self.cache[index] = item
            if self.cache_size > 0 and len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item

    def validate(self):
        hashes, min_nodes, max_nodes, total_edges = ([], math.inf, 0, 0)
        for i, graph in enumerate(self.graphs):
            raw = self.raw(i)
            x, edges = (raw["x_topology"], raw["edge_index"])
            gf, membership = (raw["graph_features"], raw["community"])
            if (
                x.ndim != 2
                or x.shape[1] != 5
                or len(x) < 2
                or (not np.isfinite(x).all())
            ):
                raise ValueError(f"Invalid topology array in {self.paths[i]}")
            n = len(x)
            if graph.get("num_nodes", n) != n:
                raise ValueError(
                    f"Node count disagrees with the manifest: {self.paths[i]}"
                )
            if gf.shape != (5,) or not np.isfinite(gf).all():
                raise ValueError(f"Invalid graph features in {self.paths[i]}")
            if (
                membership.shape != (n,)
                or membership.dtype.kind not in "iu"
                or (membership < 0).any()
            ):
                raise ValueError(f"Invalid community membership in {self.paths[i]}")
            if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
                raise ValueError(f"Invalid edge_index in {self.paths[i]}")
            if edges.size and (edges.min() < 0 or edges.max() >= n):
                raise ValueError(f"Edge endpoint out of range in {self.paths[i]}")
            if (edges[0] == edges[1]).any():
                raise ValueError(
                    f"Prepared graphs must not contain self-loops: {self.paths[i]}"
                )
            if graph.get("undirected_edges", edges.shape[1] // 2) * 2 != edges.shape[1]:
                raise ValueError(
                    f"Edge count disagrees with the manifest: {self.paths[i]}"
                )
            codes = edges[0].astype(np.int64) * n + edges[1]
            reverse = edges[1].astype(np.int64) * n + edges[0]
            if len(np.unique(codes)) != len(codes) or not np.array_equal(
                np.sort(codes), np.sort(reverse)
            ):
                raise ValueError(
                    f"Edges must be unique and bidirectional: {self.paths[i]}"
                )
            actual_hash = file_hash(self.paths[i])
            if graph.get("sha256") and graph["sha256"] != actual_hash:
                raise ValueError(f"Prepared graph SHA256 mismatch: {self.paths[i]}")
            hashes.append(
                dict(id=str(graph["id"]), label=graph["label"], sha256=actual_hash)
            )
            min_nodes, max_nodes = (min(min_nodes, n), max(max_nodes, n))
            total_edges += edges.shape[1]
        identity = dict(manifest_sha256=file_hash(self.path), graph_files=hashes)
        return dict(
            fingerprint=digest(identity),
            identity=identity,
            graphs=len(self.graphs),
            class_counts=np.bincount(
                self.labels, minlength=len(self.class_names)
            ).tolist(),
            node_count_range=[int(min_nodes), int(max_nodes)],
            directed_edge_entries=total_edges,
        )


def stratified_split(data, seed, train_fraction, val_fraction):
    rng = np.random.default_rng(seed)
    split = dict(train=[], validation=[], test=[])
    test_fraction = 1 - train_fraction - val_fraction
    for label in range(len(data.class_names)):
        indices = np.flatnonzero(data.labels == label)
        if len(indices) < 4:
            raise ValueError(
                "Each class requires at least four graphs for the three-way split"
            )
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_test = max(1, int(round(len(indices) * test_fraction)))
        n_train = len(indices) - n_val - n_test
        if n_train < 2:
            raise ValueError(
                "Each class requires at least two training graphs after rounding"
            )
        split["train"].extend(indices[:n_train].tolist())
        split["validation"].extend(indices[n_train : n_train + n_val].tolist())
        split["test"].extend(indices[n_train + n_val :].tolist())
    for key in split:
        rng.shuffle(split[key])
    flat = sum(split.values(), [])
    assert len(set(flat)) == len(data.graphs) == len(flat)
    return split


def fit_scaler(arrays, width):
    count, mean, m2 = (
        0,
        np.zeros(width, dtype=np.float64),
        np.zeros(width, dtype=np.float64),
    )
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, width)
        next_n = len(arr)
        next_mean = arr.mean(axis=0)
        next_m2 = np.square(arr - next_mean).sum(axis=0)
        delta = next_mean - mean
        total = count + next_n
        m2 += next_m2 + np.square(delta) * count * next_n / total
        mean += delta * next_n / total
        count = total
    variance = np.maximum(m2 / count, 0)
    scale = np.sqrt(variance)
    scale[scale <= 1e-12] = 1.0
    return dict(
        count=count,
        mean=mean.tolist(),
        scale=scale.tolist(),
        population_variance=variance.tolist(),
    )


def normalise(values, stats):
    return (
        (np.asarray(values, dtype=np.float64) - np.asarray(stats["mean"]))
        / np.asarray(stats["scale"])
    ).astype(np.float32)


def epoch_batches(indices, batch_size, seed=None):
    order = list(indices)
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def measure_metrics(labels, predicted, classes):
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support, selected = (confusion.sum(1), confusion.sum(0))
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive, selected, out=np.zeros(classes, dtype=float), where=selected > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(classes, dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(classes, dtype=float),
        where=precision + recall > 0,
    )
    return dict(
        accuracy=float(true_positive.sum() / confusion.sum()),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        weighted_f1=float(np.dot(f1, support) / support.sum()),
        confusion_matrix=confusion.tolist(),
        per_class=[
            dict(
                label=i,
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i in range(classes)
        ],
    )


def setup_torch(config, source_root=None):
    global torch, F, Data, DataLoader, global_mean_pool, baseline_models, make_features
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    from . import models as baseline_models
    from .features import make_features

    torch.set_num_threads(config["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True, warn_only=config["allow_nondeterministic"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    requested = config["device"]
    if requested == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def model_for(architecture, config, classes):
    base = getattr(baseline_models, "GraphFamilyClassifier" + architecture)
    return base(5, config["hidden_channels"], classes, 5, config["dropout"])


class ConditionDataset:
    def __init__(self, data, normalisation, condition, graph_on, seed):
        self.data, self.stats, self.condition = (data, normalisation, condition)
        self.graph_on, self.seed = (graph_on, seed)

    def __len__(self):
        return len(self.data.graphs)

    def __getitem__(self, index):
        raw = self.data.raw(index)
        z = normalise(raw["x_topology"], self.stats["node"])
        x = np.asarray(
            make_features(
                z,
                raw["community"],
                self.condition,
                self.seed,
                str(self.data.graphs[index]["id"]),
            ),
            dtype=np.float32,
        )
        if x.shape != z.shape or not np.isfinite(x).all():
            raise ValueError(
                f"Invalid generated features for {self.condition}, graph {index}"
            )
        gf = (
            normalise(raw["graph_features"], self.stats["graph"])
            if self.graph_on
            else np.zeros(5, dtype=np.float32)
        )
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.as_tensor(raw["edge_index"], dtype=torch.long),
            y=torch.tensor([self.data.graphs[index]["label"]], dtype=torch.long),
            graph_features=torch.from_numpy(gf.reshape(1, -1)),
            graph_idx=torch.tensor([index], dtype=torch.long),
            num_nodes=len(x),
        )


def loader_for(dataset, indices, config, shuffle_seed=None):
    generator = torch.Generator()
    generator.manual_seed(0 if shuffle_seed is None else shuffle_seed)
    return DataLoader(
        dataset,
        batch_sampler=epoch_batches(indices, config["batch_size"], shuffle_seed),
        num_workers=0,
        generator=generator,
    )


def hardware_provenance(device, config):
    software = {}
    for package in (
        "numpy",
        "torch",
        "torch-geometric",
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
    ):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu.append(
                dict(
                    index=i,
                    name=props.name,
                    total_memory=props.total_memory,
                    capability=[props.major, props.minor],
                )
            )
    return dict(
        python=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        logical_cpus=os.cpu_count(),
        hostname=socket.gethostname(),
        software=software,
        device=str(device),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpus=gpu,
        threads=config["threads"],
        deterministic_algorithms=True,
        deterministic_warn_only=config["allow_nondeterministic"],
        tf32=False,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
