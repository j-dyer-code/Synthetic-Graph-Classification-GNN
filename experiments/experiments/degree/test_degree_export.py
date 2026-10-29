import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from export_degree_histograms import (
    FAMILIES,
    NODE_NAMES,
    GRAPH_NAMES,
    sha256,
    export_histograms,
    select_manifest,
    canonical,
    relative_name,
)


def save_manifest(root, manifest):
    identity = dict(
        source_files=manifest["source_files"],
        graphs=manifest["graphs"],
        node_names=NODE_NAMES,
        graph_names=GRAPH_NAMES,
        community_seed=manifest["community_seed"],
    )
    manifest["dataset_sha256"] = sha256(json.dumps(identity, sort_keys=True).encode())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def fixture(root, per_family=1):
    root.mkdir(parents=True)
    (root / "graphs").mkdir()
    undirected = [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (1, 2), (2, 3), (0, 3)],
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    ]
    records = []
    for label, pairs in enumerate(undirected):
        edge_pairs = np.array(pairs, dtype=np.int64)
        edges = np.concatenate([edge_pairs, edge_pairs[:, ::-1]]).T.copy()
        n = int(edges.max()) + 1
        degree = np.bincount(edges[0], minlength=n)
        for repeat in range(per_family):
            gid = label * per_family + repeat
            x = np.zeros((n, 5), dtype=np.float32)
            x[:, 0] = degree
            path = root / "graphs" / f"graph_{gid}.npz"
            np.savez_compressed(
                path,
                x_topology=x,
                edge_index=edges,
                graph_features=np.zeros(5, dtype=np.float32),
                community=np.zeros(n, dtype=np.int64),
            )
            records.append(
                dict(
                    id=str(gid),
                    label=label,
                    path=path.relative_to(root).as_posix(),
                    num_nodes=n,
                    undirected_edges=len(pairs),
                    num_communities=1,
                    sha256=sha256(path.read_bytes()),
                )
            )
    manifest = dict(
        schema_version=1,
        status="complete",
        class_names=FAMILIES,
        node_feature_names=NODE_NAMES,
        graph_feature_names=GRAPH_NAMES,
        graphs=records,
        counts={family: per_family for family in FAMILIES},
        source_files=[
            dict(
                path="graph_objects/software_fixture.pkl",
                sha256=sha256(b"software fixture"),
                bytes=16,
            )
        ],
        community_seed=20261002,
        purpose="SOFTWARE TEST FIXTURE",
    )
    save_manifest(root, manifest)
    return manifest


class DegreeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.prepared = self.root / "prepared"
        self.manifest = fixture(self.prepared)
        self.output = self.root / "degrees.zip"

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        return export_histograms(self.prepared / "manifest.json", self.output, 1)

    def change_npz(self, mutate):
        record = self.manifest["graphs"][0]
        path = self.prepared / record["path"]
        with np.load(path) as data:
            values = {name: data[name] for name in data.files}
        mutate(values)
        np.savez_compressed(path, **values)
        record["sha256"] = sha256(path.read_bytes())
        save_manifest(self.prepared, self.manifest)

    def test_true_degrees_full_output_schema_and_hashes(self):
        report = self.export()
        self.assertEqual(report["graphs"], 5)
        self.assertEqual(report["nodes"], 20)
        self.assertEqual(report["undirected_edges"], 20)
        self.assertEqual(report["histogram_rows"], 7)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"degree_histograms.csv", "graph_metadata.csv", "export_manifest.json"},
            )
            hist = list(
                csv.DictReader(
                    io.StringIO(archive.read("degree_histograms.csv").decode())
                )
            )
            self.assertEqual(
                [
                    (int(r["degree"]), int(r["count"]))
                    for r in hist
                    if r["graph_id"] == "0"
                ],
                [(1, 2), (2, 2)],
            )
            meta = list(
                csv.DictReader(io.StringIO(archive.read("graph_metadata.csv").decode()))
            )
            self.assertEqual(
                meta[0]["degrees_sha256"],
                sha256(np.array([1, 2, 2, 1], dtype="<i8").tobytes()),
            )
            for name, value in report["files"].items():
                self.assertEqual(sha256(archive.read(name)), value["sha256"])
            identity = dict(
                manifest_sha256=sha256((self.prepared / "manifest.json").read_bytes()),
                graph_files=[
                    dict(id=g["id"], label=g["label"], sha256=g["sha256"])
                    for g in self.manifest["graphs"]
                ],
            )
            self.assertEqual(
                report["source_dataset_fingerprint"],
                sha256(canonical(identity).encode()),
            )

    def test_degree_mismatch_rejects_even_with_valid_npz_hash(self):
        self.change_npz(lambda x: x["x_topology"].__setitem__((0, 0), 2))
        with self.assertRaisesRegex(ValueError, "Degree disagrees"):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".degree-export-*")))

    def test_npz_hash_mismatch_rejects(self):
        path = self.prepared / self.manifest["graphs"][0]["path"]
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_asymmetric_edges_reject(self):
        self.change_npz(lambda x: x["edge_index"].__setitem__((slice(None), 3), [3, 0]))
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            self.export()

    def test_default_requires_original_graph_count(self):
        with self.assertRaisesRegex(ValueError, "2000 graphs"):
            export_histograms(self.prepared / "manifest.json", self.output)
        self.assertFalse(self.output.exists())

    def test_missing_array_fails(self):
        (self.prepared / self.manifest["graphs"][0]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.export()

    def test_discovery_and_ambiguity(self):
        self.assertEqual(
            select_manifest(project_root=self.root, expected_per_family=1),
            self.prepared / "manifest.json",
        )
        fixture(self.root / "another")
        with self.assertRaisesRegex(ValueError, "2 prepared manifests"):
            select_manifest(project_root=self.root, expected_per_family=1)
        self.assertEqual(
            select_manifest(prepared=self.prepared), self.prepared / "manifest.json"
        )

    def test_discovery_ignores_smoke_and_manifest_only_copies(self):
        complete = self.root / "complete_arrays"
        fixture(complete, per_family=2)
        copied = self.root / "results_copy"
        copied.mkdir()
        (copied / "manifest.json").write_bytes(
            (complete / "manifest.json").read_bytes()
        )
        selected = select_manifest(project_root=self.root, expected_per_family=2)
        self.assertEqual(selected, complete / "manifest.json")
        explicit = select_manifest(prepared=copied, expected_per_family=2)
        with self.assertRaisesRegex(ValueError, "Missing"):
            export_histograms(explicit, self.output, expected_per_family=2)

    def test_invalid_relative_paths_are_rejected(self):
        for value in (
            "",
            ".",
            "..",
            "../graphs/a.npz",
            "/graphs/a.npz",
            "C:graphs/a.npz",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_name(value)

    def test_output_never_overwritten(self):
        self.export()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.output.read_bytes(), before)

    def test_default_full_2000_graph_matrix(self):
        target = self.root / "full_fixture"
        fixture(target, 400)
        report = export_histograms(target / "manifest.json", self.output)
        self.assertEqual(report["graphs"], 2000)
        self.assertEqual(report["counts"], {family: 400 for family in FAMILIES})
        self.assertEqual(report["histogram_rows"], 2800)


if __name__ == "__main__":
    unittest.main()
