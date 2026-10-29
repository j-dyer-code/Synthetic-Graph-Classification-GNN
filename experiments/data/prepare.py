import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import numpy as np


def stable_seed(seed, graph_id, stream):
    payload = f"r1.3-v1\x00{int(seed)}\x00{graph_id}\x00{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


FAMILIES = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "holme_kim",
]
NODE_NAMES = ["Degree", "Pagerank", "K-Core", "Eigenvector", "Closeness"]
GRAPH_NAMES = [
    "Degree Variance",
    "Assortativity",
    "Density",
    "Clustering",
    "Average Path Length",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare graph arrays from graph_objects/ and graph_features/ batches."
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        type=Path,
        help="Contains graph_objects/ and graph_features/",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory; never overwrites an existing dataset",
    )
    parser.add_argument("--community-seed", type=int, default=20261002)
    parser.add_argument("--expected-per-family", type=int, default=400)
    args = parser.parse_args()
    try:
        import igraph as ig
    except ImportError:
        parser.error(
            "python-igraph is required to read the original graph pickles and compute memberships"
        )
    if args.expected_per_family < 1:
        parser.error("--expected-per-family must be positive")
    if args.output.exists():
        parser.error("--output must not already exist; use a fresh directory")
    raw = args.raw_data.resolve()
    inventory = []
    by_family = {}
    for family in FAMILIES:
        graph_files = sorted(
            (raw / "graph_objects").glob(f"{family}_graphs_batch_*.pkl")
        )
        if not graph_files:
            raise FileNotFoundError(f"No raw graph batches found for {family} in {raw}")
        batches = []
        for gpath in graph_files:
            suffix = gpath.name.split("_graphs_batch_", 1)[1].removesuffix(".pkl")
            fpath = (
                raw / "graph_features" / f"{family}_graph_features_batch_{suffix}.csv"
            )
            npath = (
                raw / "graph_features" / f"{family}_node_features_batch_{suffix}.pkl"
            )
            for path in (gpath, fpath, npath):
                if not path.is_file():
                    raise FileNotFoundError(path)
                inventory.append(
                    {
                        "path": str(path.relative_to(raw)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
            batches.append((gpath, fpath, npath))
        by_family[family] = batches
    args.output.mkdir(parents=True)
    (args.output / "graphs").mkdir()
    manifest = {
        "schema_version": 1,
        "status": "preparing",
        "purpose": "research",
        "class_names": FAMILIES,
        "node_feature_names": NODE_NAMES,
        "graph_feature_names": GRAPH_NAMES,
        "graphs": [],
        "source_files": inventory,
        "feature_selection": "Five fixed candidate channels",
        "edge_policy": "Both directions; no added self-loops; graph must be simple and undirected",
        "community_method": "igraph community_multilevel, unweighted, resolution=1.0; no family label input",
        "community_seed": args.community_seed,
        "igraph_version": ig.__version__,
        "numpy_version": np.__version__,
    }
    ids = set()
    counts = {f: 0 for f in FAMILIES}
    for label, family in enumerate(FAMILIES):
        for gpath, fpath, npath in by_family[family]:
            with gpath.open("rb") as f:
                graphs = pickle.load(f)
            with npath.open("rb") as f:
                node_values = pickle.load(f)
            with fpath.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if not len(graphs) == len(node_values) == len(rows):
                raise ValueError(f"Batch lengths differ: {gpath.name}")
            for graph, nodes, row in zip(graphs, node_values, rows):
                if int(row["family"]) != label:
                    raise ValueError("CSV family label inconsistent with filename")
                gid = str(int(row["graph_id"]))
                if gid in ids:
                    raise ValueError(f"Duplicate graph_id {gid}")
                ids.add(gid)
                if (
                    graph.is_directed()
                    or not graph.is_simple()
                    or (not graph.is_connected())
                ):
                    raise ValueError(
                        f"Graph {gid} must be connected, simple and undirected; no silent alteration is allowed"
                    )
                n = graph.vcount()
                if (
                    len(nodes) != n
                    or int(row["nodes"]) != n
                    or int(row["edges"]) != graph.ecount()
                ):
                    raise ValueError(f"Source metadata disagrees with graph {gid}")
                x = np.asarray(
                    [[v[name] for name in NODE_NAMES] for v in nodes], dtype=np.float32
                )
                gf = np.asarray(
                    [float(row[name]) for name in GRAPH_NAMES], dtype=np.float32
                )
                if not np.isfinite(x).all() or not np.isfinite(gf).all():
                    raise ValueError(
                        f"Nonfinite candidate features in graph {gid}; investigate source data"
                    )
                pairs = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
                edges = np.unique(
                    np.concatenate([pairs, pairs[:, ::-1]], axis=0), axis=0
                ).T.copy()
                ig.set_random_number_generator(
                    random.Random(stable_seed(args.community_seed, gid, "partition"))
                )
                membership = np.asarray(
                    graph.community_multilevel(resolution=1.0).membership,
                    dtype=np.int64,
                )
                rel = Path("graphs") / f"graph_{gid}.npz"
                dest = args.output / rel
                np.savez_compressed(
                    dest,
                    x_topology=x,
                    edge_index=edges,
                    graph_features=gf,
                    community=membership,
                )
                manifest["graphs"].append(
                    {
                        "id": gid,
                        "label": label,
                        "path": rel.as_posix(),
                        "num_nodes": n,
                        "undirected_edges": graph.ecount(),
                        "num_communities": int(np.unique(membership).size),
                        "sha256": sha256(dest),
                    }
                )
                counts[family] += 1
            print(f"Prepared {family}: {counts[family]} graphs", flush=True)
    if any((v != args.expected_per_family for v in counts.values())):
        raise ValueError(
            f"Counts {counts} differ from expected {args.expected_per_family}; output has no complete manifest"
        )
    manifest["status"] = "complete"
    manifest["counts"] = counts
    canonical = json.dumps(
        {
            "source_files": inventory,
            "graphs": manifest["graphs"],
            "node_names": NODE_NAMES,
            "graph_names": GRAPH_NAMES,
            "community_seed": args.community_seed,
        },
        sort_keys=True,
    ).encode()
    manifest["dataset_sha256"] = hashlib.sha256(canonical).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {len(ids)} graphs; {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
