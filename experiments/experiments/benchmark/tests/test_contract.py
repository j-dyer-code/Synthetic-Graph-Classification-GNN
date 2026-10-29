from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run
import numpy as np
from lib.data import PreparedDataset
from smoke import fixture
from lib.common import atomic_json, digest, file_hash, validate_dataset, verify_complete
from support import load_protocol
import support


class ContractTests(unittest.TestCase):
    def test_generated_results_are_not_part_of_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            (root / "protocol.json").write_text("{}")
            (root / "run.py").write_text("print(1)")
            (root / "lib/model.py").write_text("value = 1")
            with patch.object(support, "ROOT", root):
                before = support.package_identity()
                (root / "results").mkdir()
                (root / "results/history.json").write_text("[1,2,3]")
                self.assertEqual(before, support.package_identity())
                (root / "lib/model.py").write_text("value = 2")
                self.assertNotEqual(before, support.package_identity())

    def test_full_task_counts_and_uniqueness(self):
        args = SimpleNamespace(
            software_test=False,
            seeds=None,
            architectures=None,
            conditions=None,
            graph_modes=None,
            dataset=Path("dataset"),
            output=Path("output"),
            prepared=None,
            device="cpu",
            threads=4,
            stage="main",
        )
        main = list(run.commands(args))
        args.stage = "ablation"
        ablation = list(run.commands(args))
        self.assertEqual(len(main), 35)
        self.assertEqual(len(ablation), 480)
        self.assertEqual(len({tuple(command) for command in main + ablation}), 515)
        self.assertEqual(
            sum((Path(command[1]).name == "run_svm.py" for command in main)), 5
        )

    def test_modified_completed_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            identity = {"purpose": "software_test", "seed": 42}
            atomic_json(folder / "identity.json", identity)
            (folder / "result.txt").write_text("complete")
            atomic_json(
                folder / "complete.json",
                dict(
                    purpose="software_test",
                    fingerprint=digest(identity),
                    files={
                        name: file_hash(folder / name)
                        for name in ("identity.json", "result.txt")
                    },
                ),
            )
            verify_complete(folder)
            (folder / "result.txt").write_text("modified")
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_complete(folder)

    def test_new_corpus_requires_explicit_permission(self):
        protocol = load_protocol()["main"]
        data = SimpleNamespace(
            class_names=protocol["classes"],
            node_names=protocol["data"]["node_features"],
            graph_names=protocol["data"]["graph_features"],
            manifest={"purpose": "research"},
        )
        audit = {
            "graphs": 2000,
            "class_counts": [400] * 5,
            "identity": {"graph_files": []},
        }
        with self.assertRaisesRegex(ValueError, "published corpus"):
            validate_dataset(data, audit, protocol)
        validate_dataset(data, audit, protocol, new_corpus=True)
        data.manifest["purpose"] = "software_test"
        with self.assertRaisesRegex(ValueError, "research dataset"):
            validate_dataset(data, audit, protocol, new_corpus=True)


class DatasetValidationTests(unittest.TestCase):
    def test_fixture_arrays_match_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture(folder)
            data = PreparedDataset(folder / "manifest.json", cache_graphs=0)
            self.assertEqual(data.validate()["graphs"], 50)

    def test_mismatched_manifest_counts_are_rejected(self):
        for field, value, message in (
            ("num_nodes", 1, "Node count disagrees"),
            ("undirected_edges", 1, "Edge count disagrees"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                fixture(folder)
                data = PreparedDataset(folder / "manifest.json", cache_graphs=0)
                data.graphs[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    data.validate()

    def test_self_loops_are_rejected_even_with_valid_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture(folder)
            data = PreparedDataset(folder / "manifest.json", cache_graphs=0)
            raw = data.raw(0)
            raw["edge_index"] = np.column_stack((raw["edge_index"], [0, 0]))
            np.savez_compressed(data.paths[0], **raw)
            data.graphs[0]["sha256"] = file_hash(data.paths[0])
            with self.assertRaisesRegex(ValueError, "self-loops"):
                data.validate()


if __name__ == "__main__":
    unittest.main()
