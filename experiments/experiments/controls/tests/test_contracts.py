import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import commit, done_valid, lock, sha, source_hashes, write
from test_extension import fixture


class Contracts(unittest.TestCase):
    def test_singleton_batches_preserve_all_members(self):
        import torch
        from torch_geometric.data import Data
        from learning import batches

        for count in (2, 16, 17, 32, 33, 177, 178, 225):
            dataset = [
                Data(x=torch.zeros(2, 1), graph_idx=torch.tensor([i]))
                for i in range(count)
            ]
            order = list(reversed(range(count)))
            observed = [
                i
                for batch in batches(dataset, order, 16)
                for i in batch.graph_idx.tolist()
            ]
            self.assertEqual(observed, order)
            self.assertEqual(len(set(observed)), count)
            self.assertTrue(
                all((batch.num_graphs >= 2 for batch in batches(dataset, order, 16)))
            )

    def test_portable_paths_and_changed_graph_rejected(self):
        from learning import Graphs

        with tempfile.TemporaryDirectory() as base:
            original = fixture(Path(base) / "before")
            relocated = Path(base) / "after"
            shutil.move(original, relocated)
            dataset = Graphs(relocated, "synthetic", 0, "enhanced")
            self.assertEqual(dataset[0].num_nodes, 7)
            path = relocated / dataset.records[1]["path"]
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Input bytes changed"):
                dataset[1]

    def test_completed_queue_skips_device_initialisation(self):
        from work_queue import work

        with tempfile.TemporaryDirectory() as base:
            run = Path(base)
            write(run / "tasks.json", [dict(id="done", kind="trial", deps=[])])
            write(
                run / "manifest.json",
                dict(
                    smoke=True,
                    tasks_sha256=sha(run / "tasks.json"),
                    source_hashes=source_hashes(),
                ),
            )
            folder = run / "tasks" / "done"
            write(folder / "result.json", dict(ok=True))
            commit(folder, ["result.json"])
            with patch(
                "work_queue.preflight",
                side_effect=AssertionError("device must not be opened"),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(work(run, "cuda", smoke=True), 0)

    def test_completion_rejects_an_empty_artifact_list(self):
        with tempfile.TemporaryDirectory() as base:
            folder = Path(base)
            write(folder / "complete.json", dict(files={}))
            with self.assertRaisesRegex(RuntimeError, "at least one artifact"):
                done_valid(folder)

    def test_completed_queue_rejects_changed_artifacts(self):
        from work_queue import work

        with tempfile.TemporaryDirectory() as base:
            run = Path(base)
            write(run / "tasks.json", [dict(id="done", kind="trial", deps=[])])
            write(
                run / "manifest.json",
                dict(
                    smoke=True,
                    tasks_sha256=sha(run / "tasks.json"),
                    source_hashes=source_hashes(),
                ),
            )
            folder = run / "tasks" / "done"
            write(folder / "result.json", dict(ok=True))
            commit(folder, ["result.json"])
            write(folder / "result.json", dict(ok=False))
            with patch(
                "work_queue.preflight",
                side_effect=AssertionError("device must not be opened"),
            ):
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    work(run, "cuda", smoke=True)

    def test_reset_requires_idle_workers(self):
        from work_queue import retry_failures

        with tempfile.TemporaryDirectory() as base:
            run = Path(base)
            write(run / "tasks.json", [])
            write(
                run / "manifest.json",
                dict(
                    smoke=True,
                    tasks_sha256=sha(run / "tasks.json"),
                    source_hashes=source_hashes(),
                ),
            )
            with lock(run / "workers.lock", shared=True):
                with self.assertRaisesRegex(RuntimeError, "Stop all workers"):
                    retry_failures(run)

    def test_training_moments_reject_other_partition(self):
        from prepare import verify_moments

        moments = [
            dict(count=2, mean=np.full(5, i), m2=np.ones(5), graph=np.full(5, i))
            for i in range(3)
        ]
        valid = dict(
            count=4,
            mean=[0.5] * 5,
            population_variance=[0.75] * 5,
            scale=[np.sqrt(0.75)] * 5,
        )
        verify_moments(valid, [0, 1], moments, "node")
        with self.assertRaisesRegex(ValueError, "scaler mismatch"):
            verify_moments(valid, [1, 2], moments, "node")

    def test_rng_follows_selected_cuda_device(self):
        import torch
        from learning import rng_state, restore_rng

        value = torch.tensor([13], dtype=torch.uint8)
        with patch("torch.cuda.get_rng_state", return_value=value) as get_state:
            saved = rng_state(torch.device("cuda:0"))
            get_state.assert_called_once_with(torch.device("cuda:0"))
        with patch("torch.cuda.set_rng_state") as set_state:
            restore_rng(saved, torch.device("cuda:1"))
            set_state.assert_called_once_with(value, torch.device("cuda:1"))
        with patch(
            "torch.cuda.get_rng_state",
            side_effect=AssertionError("CPU task cannot read CUDA state"),
        ):
            self.assertIsNone(rng_state(torch.device("cpu"))["cuda"])

    def test_identity_uses_ordered_content_not_manifest_format(self):
        from prepare import SEMANTICS, synthetic

        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            graph = root / "graph.npz"
            np.savez_compressed(
                graph,
                x_topology=np.ones((2, 5)),
                graph_features=np.ones(5),
                edge_index=np.array([[0, 1], [1, 0]]),
            )
            graphs = [
                dict(id=str(i), label=i // 400, path="graph.npz", sha256=sha(graph))
                for i in range(2000)
            ]
            published = root / "published.json"
            write(
                published,
                dict(
                    graphs=[
                        {key: row[key] for key in ("id", "label", "sha256")}
                        for row in graphs
                    ]
                ),
            )
            manifest = root / "manifest.json"
            value = dict(status="complete", graphs=graphs, **SEMANTICS)
            write(manifest, value)
            with patch("prepare.PUBLISHED_IDENTITY", published):
                first = synthetic(manifest)[1]
                import json

                value["generated_at"] = "different metadata"
                manifest.write_text(json.dumps(value, indent=2))
                second = synthetic(manifest)[1]
                self.assertEqual(first, second)
                value["graphs"] = list(reversed(graphs))
                write(manifest, value)
                with self.assertRaisesRegex(ValueError, "Graph content differs"):
                    synthetic(manifest)
                value["graphs"] = graphs
                value["node_feature_names"] = list(
                    reversed(SEMANTICS["node_feature_names"])
                )
                write(manifest, value)
                with self.assertRaisesRegex(ValueError, "column order"):
                    synthetic(manifest)

    def test_split_rejects_duplicates_and_group_leakage(self):
        from prepare import validate_split

        records = [dict(label=i % 2, group=str(i)) for i in range(6)]
        split = dict(train=[0, 1], validation=[2, 3], test=[4, 5])
        validate_split(split, records, groups=True)
        with self.assertRaises(ValueError):
            validate_split(dict(split, train=[0, 1, 1]), records)
        records[2]["group"] = "0"
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_split(split, records, groups=True)

    def test_zero_variance_contrasts_do_not_claim_a_p_value(self):
        from analyse import contrast_statistics

        for differences in ([0.0] * 5, [0.25] * 5, [0.1] * 5):
            result = contrast_statistics(differences, [0.2] * 5)
            self.assertTrue(result["degenerate_variance"])
            self.assertIsNone(result["uncorrected_p"])
            self.assertIsNone(result["bonferroni_95_lower_pp"])
            self.assertIsNone(result["bonferroni_95_upper_pp"])

    def test_nonconstant_contrast_retains_corrected_inference(self):
        from analyse import contrast_statistics
        from scipy.stats import t

        differences = np.array([0.01, 0.02, 0.04, 0.03, 0.05])
        ratios = np.array([0.2, 0.25, 0.3, 0.2, 0.25])
        result = contrast_statistics(differences, ratios)
        se = np.sqrt((1 / 5 + ratios.mean()) * differences.var(ddof=1))
        critical = t.ppf(1 - 0.05 / (2 * 24), 4)
        self.assertFalse(result["degenerate_variance"])
        self.assertAlmostEqual(result["corrected_se_pp"], 100 * se)
        self.assertAlmostEqual(
            result["uncorrected_p"], 2 * t.sf(abs(differences.mean() / se), 4)
        )
        self.assertAlmostEqual(
            result["bonferroni_95_lower_pp"], 100 * (differences.mean() - critical * se)
        )
        self.assertAlmostEqual(
            result["bonferroni_95_upper_pp"], 100 * (differences.mean() + critical * se)
        )

    def test_missing_p_values_preserve_the_prespecified_family(self):
        from analyse import adjust_contrast_p_values

        rows = [dict(uncorrected_p=value) for value in [0.04, None, 0.01] + [None] * 21]
        adjust_contrast_p_values(rows)
        self.assertAlmostEqual(rows[2]["holm_adjusted_p"], 0.24)
        self.assertAlmostEqual(rows[0]["holm_adjusted_p"], 0.92)
        self.assertTrue(
            all(
                row["holm_adjusted_p"] is None
                for row in rows
                if row["uncorrected_p"] is None
            )
        )

    def test_holm_adjustment_remains_monotone(self):
        from analyse import adjust_contrast_p_values

        values = [0.01, 0.0101, 0.02] + [0.5] * 21
        rows = [dict(uncorrected_p=value) for value in values]
        adjust_contrast_p_values(rows)
        adjusted = [row["holm_adjusted_p"] for row in rows]
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertAlmostEqual(adjusted[0], 0.24)
        self.assertAlmostEqual(adjusted[1], 0.24)
        self.assertAlmostEqual(adjusted[2], 0.44)


if __name__ == "__main__":
    unittest.main(verbosity=2)
