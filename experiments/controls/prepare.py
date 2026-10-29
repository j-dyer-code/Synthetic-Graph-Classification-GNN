import argparse
import collections
import io
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from core import (
    ROOT,
    PUBLISHED_IDENTITY,
    read,
    write,
    sha,
    digest,
    source_hashes,
    lock,
    done_valid,
    data_path,
)
from protocol import REAL, SETTINGS, tasks


def download(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in REAL:
        p = dest / (name + ".zip")
        with lock(dest / (name + ".download.lock"), blocking=True):
            if not p.exists():
                url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
                print("Downloading", url, flush=True)
                temporary = Path(str(p) + ".tmp")
                try:
                    with urllib.request.urlopen(url, timeout=180) as r, temporary.open(
                        "wb"
                    ) as f:
                        shutil.copyfileobj(r, f)
                    if (
                        sha(temporary)
                        != read(ROOT / "public_data_checksums.json")[name]
                    ):
                        raise ValueError(
                            "Downloaded archive differs from the fixed dataset version: "
                            + name
                        )
                    os.replace(temporary, p)
                finally:
                    temporary.unlink(missing_ok=True)
            if sha(p) != read(ROOT / "public_data_checksums.json")[name]:
                raise ValueError("Dataset archive checksum mismatch: " + str(p))
            with zipfile.ZipFile(p) as z:
                if z.testzip():
                    raise ValueError("Bad TU archive: " + str(p))
        print(name, sha(p), flush=True)


def attributed_group(n, edge, x):
    adj = [[] for _ in range(n)]
    for u, v in edge.T:
        adj[int(u)].append(int(v))
    labels = [digest([float(v) for v in row]) for row in x]
    for _ in range(5):
        labels = [
            digest([labels[u], sorted((labels[v] for v in adj[u]))]) for u in range(n)
        ]
    return digest([n, edge.shape[1], sorted(labels)])


def descriptors(n, edge, seed):
    import igraph as ig

    ig.set_random_number_generator(random.Random(seed))
    undirected = [(int(a), int(b)) for a, b in edge.T if a < b]
    g = ig.Graph(n=n, edges=undirected, directed=False)
    if n == 0:
        raise ValueError("Empty graph")
    x = np.array(
        [
            g.degree(),
            g.pagerank(),
            g.coreness(),
            g.eigenvector_centrality(),
            g.closeness(),
        ],
        dtype=np.float64,
    ).T
    gf = np.array(
        [
            np.var(g.degree()),
            g.assortativity_degree(directed=False),
            g.density(),
            g.transitivity_undirected(),
            g.average_path_length(directed=False, unconn=True),
        ],
        dtype=np.float64,
    )
    invalid = {
        "node": int((~np.isfinite(x)).sum()),
        "graph": int((~np.isfinite(gf)).sum()),
    }
    return (
        np.nan_to_num(x, nan=0, posinf=0, neginf=0),
        np.nan_to_num(gf, nan=0, posinf=0, neginf=0),
        invalid,
    )


def parse_tu(archive, name, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:

        def load(suffix, required=True):
            hits = [
                p
                for p in z.namelist()
                if p.endswith("/" + name + "_" + suffix + ".txt")
                or p == name + "_" + suffix + ".txt"
            ]
            if not hits:
                if required:
                    raise ValueError("TU field absent: " + suffix)
                return None
            if len(hits) != 1:
                raise ValueError("Ambiguous TU field")
            return np.loadtxt(io.BytesIO(z.read(hits[0])), delimiter=",", ndmin=2)

        indicator = load("graph_indicator").astype(int).ravel() - 1
        labels = load("graph_labels").ravel()
        vocab = sorted(set(labels.tolist()))
        target = np.array([vocab.index(v) for v in labels], dtype=int)
        edges = load("A").astype(int) - 1
        native = load("node_attributes", False)
        cats = load("node_labels", False)
        if cats is not None:
            onehot = np.concatenate(
                [
                    (cats[:, c, None] == np.unique(cats[:, c])).astype(float)
                    for c in range(cats.shape[1])
                ],
                axis=1,
            )
            native = onehot if native is None else np.c_[native, onehot]
        unattributed = native is None
        if native is None:
            native = np.ones((len(indicator), 1), dtype=float)
        if not np.isfinite(native).all():
            raise ValueError(
                "Nonfinite native features require an explicit imputation protocol"
            )
        if len(native) != len(indicator):
            raise ValueError("Node count mismatch")
        if not np.array_equal(indicator[edges[:, 0]], indicator[edges[:, 1]]):
            raise ValueError("Cross-graph TU edges")
        bygraph = [[] for _ in labels]
        for a, b in edges:
            bygraph[indicator[a]].append((int(a), int(b)))
        records = []
        audit = []
        for i, y in enumerate(target):
            nodes = np.flatnonzero(indicator == i)
            lut = {int(v): j for j, v in enumerate(nodes)}
            pairs = set()
            for a, b in bygraph[i]:
                a, b = (lut[a], lut[b])
                if a != b:
                    pairs.update([(a, b), (b, a)])
            edge = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T
            x = native[nodes]
            if unattributed:
                degree = np.bincount(edge[0], minlength=len(nodes))
                x = np.c_[np.ones(len(nodes)), np.log1p(degree)]
            topo, gf, invalid = descriptors(len(nodes), edge, 100000 + i)
            p = out / f"{i}.npz"
            np.savez_compressed(
                p, edge_index=edge, x_native=x, x_topology=topo, graph_features=gf
            )
            records.append(
                dict(
                    id=str(i),
                    label=int(y),
                    path=str(p.resolve()),
                    sha256=sha(p),
                    group=attributed_group(len(nodes), edge, x),
                )
            )
            audit.append(
                dict(
                    id=str(i),
                    nodes=len(nodes),
                    edges=edge.shape[1] // 2,
                    unattributed_degree_baseline=unattributed,
                    nonfinite_replaced=invalid,
                )
            )
    write(out / "feature_audit.json", audit)
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[r["group"]].append(r["label"])
    write(
        out / "duplicates.json",
        dict(
            groups=len(grouped),
            graphs=len(records),
            nonunique_groups=sum((len(v) > 1 for v in grouped.values())),
            conflicting_label_groups=sum((len(set(v)) > 1 for v in grouped.values())),
            method=SETTINGS["duplicate_groups"],
        ),
    )
    return records


def scaler(arrays):
    count = 0
    mean = None
    m2 = None
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        a = a.reshape(-1, a.shape[-1])
        n = len(a)
        am = a.mean(0)
        av = ((a - am) ** 2).sum(0)
        if mean is None:
            count, mean, m2 = (n, am, av)
        else:
            delta = am - mean
            total = count + n
            m2 += av + delta**2 * count * n / total
            mean += delta * n / total
            count = total
    s = np.sqrt(m2 / count)
    s[s <= 1e-12] = 1
    return dict(mean=mean.tolist(), scale=s.tolist(), count=count)


def validate_split(split, records, groups=False):
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names")
    for name, part in split.items():
        if (
            len(part) < 2
            or len(part) != len(set(part))
            or any((type(i) is not int or i < 0 or i >= len(records) for i in part))
        ):
            raise ValueError("Invalid or duplicated split membership: " + name)
    partitions = [set(split[k]) for k in ("train", "validation", "test")]
    if any((partitions[a] & partitions[b] for a in range(3) for b in range(a))):
        raise ValueError("Split overlap")
    if set.union(*partitions) != set(range(len(records))):
        raise ValueError("Split missing graph")
    classes = set((r["label"] for r in records))
    for part in partitions:
        if set((records[i]["label"] for i in part)) != classes:
            raise ValueError("A split lacks a class")
    if groups:
        gs = [{records[i]["group"] for i in p} for p in partitions]
        if any((gs[a] & gs[b] for a in range(3) for b in range(a))):
            raise ValueError("Duplicate group leakage")


SEMANTICS = dict(
    class_names=[
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
        "stochastic_block_model",
        "holme_kim",
    ],
    node_feature_names=["Degree", "Pagerank", "K-Core", "Eigenvector", "Closeness"],
    graph_feature_names=[
        "Degree Variance",
        "Assortativity",
        "Density",
        "Clustering",
        "Average Path Length",
    ],
)


def synthetic(source, new_corpus=False):
    source = Path(source).resolve()
    manifest = read(source)
    if manifest.get("status") != "complete":
        raise ValueError("Synthetic preparation is incomplete")
    for key, value in SEMANTICS.items():
        if manifest.get(key) != value:
            raise ValueError("Synthetic semantic column order differs: " + key)
    records = []
    content = []
    moments = []
    for record in manifest["graphs"]:
        relative = Path(record["path"])
        path = (source.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(source.parent):
            raise ValueError("Synthetic paths must stay within the manifest directory")
        graph_id = str(record["id"])
        label = record["label"]
        actual = sha(path)
        if type(label) is not int or not 0 <= label < 5:
            raise ValueError("Invalid synthetic class label")
        if actual != record.get("sha256"):
            raise ValueError("Synthetic graph checksum mismatch: " + graph_id)
        with np.load(path, allow_pickle=False) as arrays:
            x = np.asarray(arrays["x_topology"], dtype=np.float64)
            graph = np.asarray(arrays["graph_features"], dtype=np.float64)
            edges = arrays["edge_index"]
            if x.ndim != 2 or x.shape[1] != 5 or len(x) < 1 or (graph.shape != (5,)):
                raise ValueError("Invalid synthetic feature dimensions: " + graph_id)
            if not np.isfinite(x).all() or not np.isfinite(graph).all():
                raise ValueError("Nonfinite synthetic features")
            if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
                raise ValueError("Invalid edge array")
            if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
                raise ValueError("Edge index out of bounds")
            mean = x.mean(0)
            moments.append(
                dict(
                    count=len(x),
                    mean=mean,
                    m2=((x - mean) ** 2).sum(0),
                    graph=graph.copy(),
                )
            )
        records.append(dict(id=graph_id, label=label, path=str(path), sha256=actual))
        content.append(dict(id=graph_id, label=label, sha256=actual))
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate graph IDs")
    if collections.Counter(
        (record["label"] for record in records)
    ) != collections.Counter({i: 400 for i in range(5)}):
        raise ValueError(
            "The research protocol requires 400 graphs for each of five families"
        )
    published = read(PUBLISHED_IDENTITY)
    matches = content == published["graphs"]
    if not new_corpus and (not matches):
        raise ValueError(
            "Graph content differs from the published corpus. To run a new experiment, use --new-corpus with --prepared from benchmark/prepare.py; frozen transforms cannot be reused."
        )
    identity = dict(
        schema="ordered-graph-content-v1",
        scope="new_corpus" if new_corpus else "published",
        matches_published_graphs=matches,
        content_sha256=digest(dict(**SEMANTICS, graph_files=content)),
        graph_files=content,
        **SEMANTICS,
    )
    return (records, identity, moments)


def checked_preparation(prepared, identity):
    prepared = Path(prepared).resolve()
    if not done_valid(prepared, True):
        raise ValueError("Benchmark preparation is incomplete")
    marker = read(prepared / "complete.json")
    metadata = read(prepared / "identity.json")
    audit = read(prepared / "dataset_audit.json")
    if marker["fingerprint"] != digest(metadata):
        raise ValueError("Benchmark preparation identity mismatch")
    if (
        metadata.get("purpose") != "research"
        or metadata.get("dataset_scope") != "new_corpus"
    ):
        raise ValueError(
            "--new-corpus requires a research preparation made with benchmark/prepare.py --new-corpus"
        )
    for actual in (
        audit["identity"]["graph_files"],
        metadata["dataset_identity"]["graph_files"],
    ):
        if actual != identity["graph_files"]:
            raise ValueError("Prepared selectors belong to different graph content")
    protocol = read(prepared / "protocol.json")
    if protocol["seeds"] != list(range(42, 47)):
        raise ValueError("Shared preparation must contain all five seeds 42–46")
    return (prepared, metadata)


def verify_moments(stats, indices, moments, level):
    if level == "graph":
        values = np.array([moments[i]["graph"] for i in indices])
        mean = values.mean(0)
        variance = ((values - mean) ** 2).mean(0)
        count = len(values)
    else:
        count = 0
        mean = np.zeros(5)
        m2 = np.zeros(5)
        for i in indices:
            row = moments[i]
            n = row["count"]
            total = count + n
            delta = row["mean"] - mean
            m2 += row["m2"] + delta**2 * count * n / total
            mean += delta * n / total
            count = total
        variance = m2 / count
    scale = np.sqrt(variance)
    scale[scale <= 1e-12] = 1
    if stats["count"] != count:
        raise ValueError("Scaler observation count differs from the training partition")
    for key, actual in [
        ("mean", mean),
        ("population_variance", variance),
        ("scale", scale),
    ]:
        saved = np.asarray(stats[key], dtype=np.float64)
        if (
            saved.shape != (5,)
            or not np.isfinite(saved).all()
            or (not np.allclose(saved, actual, rtol=1e-09, atol=1e-12))
        ):
            raise ValueError(f"Training-only {level} scaler mismatch: {key}")


def synthetic_fold(reference, records, moments, expected_fingerprint):
    split_record = read(reference / "split.json")
    norm = read(reference / "normalisation.json")
    selection = read(reference / "selection.json")
    if split_record["dataset_fingerprint"] != expected_fingerprint:
        raise ValueError("Split dataset identity differs")
    index = {record["id"]: i for i, record in enumerate(records)}
    split = {
        name: [index[str(v)] for v in ids] for name, ids in split_record["ids"].items()
    }
    validate_split(split, records)
    training = [records[i]["id"] for i in split["train"]]
    if (
        norm["fit_training_graph_ids"] != training
        or selection["training_graph_ids"] != training
    ):
        raise ValueError(
            "Selectors/scalers were not fitted to the exact training partition"
        )
    stats = {level: norm[level] for level in ["node", "graph"]}
    selected = {
        level: selection[level]["selected_indices"] for level in ["node", "graph"]
    }
    for level in stats:
        columns = selected[level]
        if (
            not columns
            or len(columns) != len(set(columns))
            or any((type(i) is not int or not 0 <= i < 5 for i in columns))
        ):
            raise ValueError("Invalid selected feature order")
        verify_moments(stats[level], split["train"], moments, level)
    return (split, stats, selected)


def prepare(args):
    if args.new_corpus and (not args.prepared):
        raise ValueError("--new-corpus requires --prepared")
    if args.prepared and (not args.new_corpus):
        raise ValueError(
            "--prepared is for explicit new-corpus runs; published runs use the frozen reference preparation"
        )
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with lock(run / "prepare.lock") as held:
        if held is None:
            raise RuntimeError("Preparation is already running")
        if (run / "manifest.json").exists():
            raise RuntimeError("Run already prepared; choose a new run directory")
        records, identity, moments = synthetic(args.synthetic, args.new_corpus)
        prepared, shared = (
            checked_preparation(args.prepared, identity)
            if args.new_corpus
            else (ROOT / "reference_splits", None)
        )
        if shared is not None:
            identity["shared_preparation_sha256"] = digest(shared)
        synthetic_records = records
        for dataset in ["synthetic", *REAL]:
            folder = run / "data" / dataset
            folder.mkdir(parents=True, exist_ok=True)
            if dataset == "synthetic":
                records = synthetic_records
                write(folder / "identity.json", identity)
            else:
                archive = Path(args.downloads) / (dataset + ".zip")
                if sha(archive) != read(ROOT / "public_data_checksums.json")[dataset]:
                    raise ValueError("Public dataset archive differs: " + dataset)
                records = parse_tu(archive, dataset, folder / "graphs")
            records = [
                dict(record, path=os.path.relpath(record["path"], run))
                for record in records
            ]
            write(folder / "records.json", records)
            labels = np.array([record["label"] for record in records])
            if dataset != "synthetic":
                from sklearn.model_selection import StratifiedGroupKFold

                groups = np.array([record["group"] for record in records])
                outer = list(
                    StratifiedGroupKFold(5, shuffle=True, random_state=5102026).split(
                        np.zeros(len(labels)), labels, groups
                    )
                )
            for fold in range(5):
                if dataset == "synthetic":
                    reference = prepared / f"seed_{42 + fold}"
                    fingerprint = (
                        shared["dataset_fingerprint"]
                        if shared
                        else read(PUBLISHED_IDENTITY)["fingerprint"]
                    )
                    split, stats, selected = synthetic_fold(
                        reference, records, moments, fingerprint
                    )
                    for name in ["split.json", "normalisation.json", "selection.json"]:
                        write(
                            folder / "preparation" / f"seed_{42 + fold}" / name,
                            read(reference / name),
                        )
                else:
                    development, test = outer[fold]
                    train, val = next(
                        StratifiedGroupKFold(
                            5, shuffle=True, random_state=5102100 + fold
                        ).split(development, labels[development], groups[development])
                    )
                    split = dict(
                        train=development[train].tolist(),
                        validation=development[val].tolist(),
                        test=test.tolist(),
                    )

                    def arrays(field):
                        for i in split["train"]:
                            with np.load(
                                data_path(run, records[i]), allow_pickle=False
                            ) as values:
                                yield values[field].copy()

                    stats = {
                        key: scaler(arrays(field))
                        for key, field in [
                            ("node", "x_topology"),
                            ("graph", "graph_features"),
                            ("native", "x_native"),
                        ]
                    }
                    selected = dict(node=list(range(5)), graph=list(range(5)))
                validate_split(split, records, dataset != "synthetic")
                write(
                    folder / f"fold_{fold}.json",
                    dict(
                        split=split,
                        stats=stats,
                        selected=selected,
                        training_ids=[records[i]["id"] for i in split["train"]],
                        class_counts={
                            name: np.bincount(labels[indices]).tolist()
                            for name, indices in split.items()
                        },
                    ),
                )
            print("Prepared", dataset, len(records), "graphs", flush=True)
        task_list = tasks()
        write(run / "tasks.json", task_list)
        manifest = dict(
            schema=1,
            protocol=SETTINGS,
            source_hashes=source_hashes(),
            tasks_sha256=sha(run / "tasks.json"),
            task_count=len(task_list),
            smoke=False,
            synthetic_scope=identity["scope"],
            synthetic_content_sha256=identity["content_sha256"],
            data_hashes={
                str(path.relative_to(run)): sha(path)
                for path in sorted((run / "data").rglob("*.json"))
            },
            tu_archives={
                name: sha(Path(args.downloads) / (name + ".zip")) for name in REAL
            },
        )
        write(run / "manifest.json", manifest)
        print(
            "READY:",
            len(task_list),
            "tasks;",
            identity["scope"],
            "synthetic corpus;",
            run,
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download application datasets and prepare matched-control experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download")
    command.add_argument("--downloads", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--run", required=True)
    command.add_argument("--synthetic", required=True)
    command.add_argument("--downloads", required=True)
    command.add_argument("--new-corpus", action="store_true")
    command.add_argument(
        "--prepared", help="Shared benchmark preparation for the new corpus"
    )
    args = parser.parse_args()
    download(args.downloads) if args.command == "download" else prepare(args)
