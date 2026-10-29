import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile
import numpy as np

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
HEX = re.compile("^[0-9a-f]{64}$")
HIST_COLUMNS = ["graph_id", "family", "degree", "count"]
META_COLUMNS = [
    "graph_id",
    "family",
    "num_nodes",
    "num_edges",
    "degree_sum",
    "min_degree",
    "max_degree",
    "mean_degree",
    "distinct_degrees",
    "degrees_sha256",
]


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Source paths must be nonempty portable relative paths")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or ":" in path.parts[0]
    ):
        raise ValueError("Absolute or parent-traversing source paths are not accepted")
    return path


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def select_manifest(prepared=None, project_root=None, expected_per_family=400):
    if prepared is not None:
        path = Path(prepared).expanduser().resolve()
        path = path / "manifest.json" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError("The specified prepared manifest does not exist")
        return path
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError("The project root is not an existing directory")
    integer(expected_per_family, "expected_per_family", 1)
    expected_counts = {family: expected_per_family for family in FAMILIES}
    candidates = []
    for path in root.rglob("manifest.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(value, dict)
            and value.get("schema_version") == 1
            and (value.get("status") == "complete")
            and (value.get("class_names") == FAMILIES)
            and (value.get("node_feature_names") == NODE_NAMES)
            and isinstance(value.get("graphs"), list)
            and (len(value["graphs"]) == len(FAMILIES) * expected_per_family)
            and (value.get("counts") == expected_counts)
        ):
            try:
                dataset_root = path.parent.resolve()
                labels = Counter((graph["label"] for graph in value["graphs"]))
                if labels != Counter(
                    {label: expected_per_family for label in range(len(FAMILIES))}
                ):
                    continue
                present = True
                for graph in value["graphs"]:
                    array_path = (
                        dataset_root / str(relative_name(graph["path"]))
                    ).resolve()
                    if (
                        not array_path.is_relative_to(dataset_root)
                        or not array_path.is_file()
                    ):
                        present = False
                        break
                if present:
                    candidates.append(path.resolve())
            except (KeyError, TypeError, ValueError, OSError):
                continue
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No prepared graph manifest found; use --prepared to specify it"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Found {len(candidates)} prepared manifests; select one explicitly with --prepared"
        )
    return candidates[0]


def read_manifest(path, expected_per_family):
    integer(expected_per_family, "expected_per_family", 1)
    blob = path.read_bytes()
    manifest = json.loads(blob)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Expected a complete schema-1 prepared manifest")
    if manifest.get("class_names") != FAMILIES:
        raise ValueError(
            "The manifest must declare the five family labels in protocol order"
        )
    if (
        manifest.get("node_feature_names") != NODE_NAMES
        or manifest.get("graph_feature_names") != GRAPH_NAMES
    ):
        raise ValueError("The five node and graph descriptor names are required")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != 5 * expected_per_family:
        raise ValueError(f"Expected {5 * expected_per_family} graphs in total")
    ids, paths, counts = (set(), set(), Counter())
    root = path.parent.resolve()
    for graph in graphs:
        gid = str(graph["id"])
        if not gid.isdigit() or str(int(gid)) != gid or gid in ids:
            raise ValueError("Graph IDs must be unique canonical nonnegative integers")
        ids.add(gid)
        label = integer(graph["label"], "graph label")
        if label >= len(FAMILIES):
            raise ValueError("Graph label is outside the declared family mapping")
        counts[FAMILIES[label]] += 1
        rel = relative_name(graph["path"])
        dest = (root / str(rel)).resolve()
        if not dest.is_relative_to(root) or not dest.is_file() or dest in paths:
            raise ValueError(
                f"Missing, duplicated or out-of-dataset NPZ for graph {gid}"
            )
        paths.add(dest)
        if not isinstance(graph.get("sha256"), str) or not HEX.fullmatch(
            graph["sha256"]
        ):
            raise ValueError(f"A valid NPZ SHA-256 is required for graph {gid}")
        integer(graph.get("num_nodes"), "num_nodes", 2)
        integer(graph.get("undirected_edges"), "undirected_edges")
        integer(graph.get("num_communities"), "num_communities", 1)
    expected = {family: expected_per_family for family in FAMILIES}
    if dict(counts) != expected or manifest.get("counts") != expected:
        raise ValueError(
            "Per-family graph counts disagree with the expected complete dataset"
        )
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Declared original source-file checksums are required")
    source_names = set()
    for item in sources:
        name = str(relative_name(item["path"]))
        if name in source_names or not HEX.fullmatch(item.get("sha256", "")):
            raise ValueError(
                "Original source-file paths or SHA-256 declarations are invalid"
            )
        source_names.add(name)
        integer(item.get("bytes"), "original source-file byte count", 1)
    identity = {
        "source_files": sources,
        "graphs": graphs,
        "node_names": NODE_NAMES,
        "graph_names": GRAPH_NAMES,
        "community_seed": manifest.get("community_seed"),
    }
    expected_digest = sha256(json.dumps(identity, sort_keys=True).encode())
    if manifest.get("dataset_sha256") != expected_digest:
        raise ValueError(
            "Prepared manifest dataset_sha256 does not match its declared contents"
        )
    return (manifest, sha256(blob))


def graph_degrees(path, record):
    blob = path.read_bytes()
    if sha256(blob) != record["sha256"]:
        raise ValueError(f"NPZ SHA-256 mismatch for graph {record['id']}")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        required = {"x_topology", "edge_index", "graph_features", "community"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing prepared NPZ arrays for graph {record['id']}")
        x, edges = (data["x_topology"], data["edge_index"])
        graph_features, community = (data["graph_features"], data["community"])
    n = record["num_nodes"]
    if x.shape != (n, 5) or x.dtype.kind not in "fiu" or (not np.isfinite(x).all()):
        raise ValueError(
            f"Invalid five-channel node descriptors for graph {record['id']}"
        )
    if (
        graph_features.shape != (5,)
        or graph_features.dtype.kind not in "fiu"
        or (not np.isfinite(graph_features).all())
    ):
        raise ValueError(f"Invalid graph descriptors for graph {record['id']}")
    if (
        community.shape != (n,)
        or community.dtype.kind not in "iu"
        or (community < 0).any()
        or (len(np.unique(community)) != record["num_communities"])
    ):
        raise ValueError(f"Invalid community membership for graph {record['id']}")
    values = x[:, 0]
    if (
        (values < 0).any()
        or (values >= n).any()
        or (not np.array_equal(values, np.floor(values)))
    ):
        raise ValueError(
            f"Degree must contain nonnegative simple-graph integers for graph {record['id']}"
        )
    degrees = values.astype(np.int64)
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or edges.dtype.kind not in "iu"
        or (edges.shape[1] != 2 * record["undirected_edges"])
    ):
        raise ValueError(f"Invalid edge shape/count for graph {record['id']}")
    if edges.size and ((edges < 0).any() or (edges >= n).any()):
        raise ValueError(f"Edge endpoint outside node range for graph {record['id']}")
    edges = edges.astype(np.int64, copy=False)
    if (edges[0] == edges[1]).any():
        raise ValueError(f"Unexpected self-loop for graph {record['id']}")
    codes = edges[0] * n + edges[1]
    reverse = edges[1] * n + edges[0]
    if len(np.unique(codes)) != len(codes) or not np.array_equal(
        np.sort(codes), np.sort(reverse)
    ):
        raise ValueError(
            f"Edges must be unique and bidirectional for graph {record['id']}"
        )
    actual = np.bincount(edges[0], minlength=n)
    if not np.array_equal(degrees, actual):
        raise ValueError(
            f"Stored Degree disagrees with edge incidence for graph {record['id']}"
        )
    return degrees


def csv_bytes(columns, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def export_histograms(manifest_path, output, expected_per_family=400, progress=None):
    manifest_path, output = (Path(manifest_path).resolve(), Path(output).resolve())
    if output.exists():
        raise FileExistsError("The output already exists; choose a new ZIP filename")
    if output.suffix.lower() != ".zip":
        raise ValueError("The output filename must end in .zip")
    manifest, manifest_hash = read_manifest(manifest_path, expected_per_family)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(
        prefix=".degree-export-", suffix=".zip", dir=output.parent
    )
    os.close(descriptor)
    temp = Path(temp)
    metadata, prepared_hashes = ([], [])
    total_nodes = total_edges = histogram_rows = 0
    histogram_hash = hashlib.sha256()
    try:
        with zipfile.ZipFile(
            temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            with archive.open("degree_histograms.csv", "w") as member:
                header = (",".join(HIST_COLUMNS) + "\n").encode()
                member.write(header)
                histogram_hash.update(header)
                for index, graph in enumerate(manifest["graphs"]):
                    degrees = graph_degrees(manifest_path.parent / graph["path"], graph)
                    gid, family = (str(graph["id"]), FAMILIES[graph["label"]])
                    unique, counts = np.unique(degrees, return_counts=True)
                    rows = [
                        {
                            "graph_id": gid,
                            "family": family,
                            "degree": int(d),
                            "count": int(c),
                        }
                        for d, c in zip(unique, counts)
                    ]
                    chunk = csv_bytes(HIST_COLUMNS, rows).split(b"\n", 1)[1]
                    member.write(chunk)
                    histogram_hash.update(chunk)
                    histogram_rows += len(rows)
                    metadata.append(
                        dict(
                            graph_id=gid,
                            family=family,
                            num_nodes=len(degrees),
                            num_edges=graph["undirected_edges"],
                            degree_sum=int(degrees.sum()),
                            min_degree=int(degrees.min()),
                            max_degree=int(degrees.max()),
                            mean_degree=float(degrees.mean()),
                            distinct_degrees=len(unique),
                            degrees_sha256=sha256(
                                degrees.astype("<i8").tobytes(order="C")
                            ),
                        )
                    )
                    prepared_hashes.append(
                        dict(id=gid, label=graph["label"], sha256=graph["sha256"])
                    )
                    total_nodes += len(degrees)
                    total_edges += graph["undirected_edges"]
                    if progress and (
                        (index + 1) % 100 == 0 or index + 1 == len(manifest["graphs"])
                    ):
                        progress(index + 1, len(manifest["graphs"]))
            metadata_blob = csv_bytes(META_COLUMNS, metadata)
            archive.writestr("graph_metadata.csv", metadata_blob)
            source_identity = dict(
                manifest_sha256=manifest_hash, graph_files=prepared_hashes
            )
            report = dict(
                schema_version=1,
                status="complete",
                created_utc=datetime.now(timezone.utc).isoformat(),
                purpose="Per-graph empirical degree distributions from verified prepared graph arrays",
                graphs=len(metadata),
                expected_per_family=expected_per_family,
                class_names=FAMILIES,
                counts=manifest["counts"],
                nodes=total_nodes,
                undirected_edges=total_edges,
                histogram_rows=histogram_rows,
                input_manifest_sha256=manifest_hash,
                source_dataset_purpose=manifest.get(
                    "purpose", "Original prepared five-family graph dataset"
                ),
                source_dataset_fingerprint=sha256(canonical(source_identity).encode()),
                source_dataset_sha256=manifest["dataset_sha256"],
                source_prepared_graph_checksums=prepared_hashes,
                declared_original_source_files=manifest["source_files"],
                source_checks="Prepared NPZ SHA-256 hashes verified against the manifest.",
                degree_checks="Integer stored Degree equals outgoing edge incidence for every node; unique bidirectional loop-free edges; edge and node counts match the prepared manifest.",
                histogram_definition="One row for every degree present in a graph. Unlisted degrees have count zero. Counts sum to num_nodes separately for each graph.",
                degrees_sha256_encoding="Original node order; signed 64-bit little-endian degree integers; contiguous C-order bytes; no header.",
                files={
                    "degree_histograms.csv": {"sha256": histogram_hash.hexdigest()},
                    "graph_metadata.csv": {"sha256": sha256(metadata_blob)},
                },
                exporter_sha256=sha256(Path(__file__).read_bytes()),
                software={
                    "python": ".".join(map(str, sys.version_info[:3])),
                    "numpy": np.__version__,
                },
            )
            archive.writestr(
                "export_manifest.json",
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temp, output)
        return report
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export verified per-graph degree histograms from prepared graph arrays."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prepared", type=Path, help="Prepared directory or its manifest.json"
    )
    source.add_argument(
        "--project-root",
        type=Path,
        help="Search recursively for one prepared graph manifest",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output ZIP; never overwritten"
    )
    parser.add_argument(
        "--expected-per-family",
        type=int,
        default=400,
        help="Expected graph count per family (default: 400)",
    )
    args = parser.parse_args()
    try:
        manifest = select_manifest(
            args.prepared, args.project_root, args.expected_per_family
        )
        report = export_histograms(
            manifest,
            args.output,
            args.expected_per_family,
            progress=lambda done, total: print(
                f"Verified {done}/{total} graphs", flush=True
            ),
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(
        f"Complete: {report['graphs']} graphs; {report['histogram_rows']} histogram rows; {args.output.stat().st_size:,} ZIP bytes; output: {args.output}"
    )


if __name__ == "__main__":
    main()
