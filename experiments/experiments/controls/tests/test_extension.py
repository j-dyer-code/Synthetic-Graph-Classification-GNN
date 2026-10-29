import contextlib
import io
import multiprocessing
import os
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import write, read, sha, source_hashes, commit
from protocol import tasks, CONFIGS, ARCH_SYN


def fixture(root):
    import numpy as np

    p = Path(root)
    d = p / "data" / "synthetic"
    d.mkdir(parents=True)
    rec = []
    rng = np.random.default_rng(87)
    for i in range(40):
        n = 7
        edges = np.array(
            [
                (u, v)
                for u in range(n)
                for v in range(n)
                if u != v and (u + v + i) % 3 == 0
            ],
            dtype=np.int64,
        ).T
        f = d / f"{i}.npz"
        np.savez_compressed(
            f,
            x_topology=rng.normal(size=(n, 5)),
            graph_features=rng.normal(size=5),
            edge_index=edges,
        )
        rec.append(
            dict(id=str(i), label=i % 2, path=str(f.relative_to(p)), sha256=sha(f))
        )
    write(d / "records.json", rec)
    cfg = dict(
        training_ids=[str(i) for i in range(24)],
        split=dict(
            train=list(range(24)),
            validation=list(range(24, 32)),
            test=list(range(32, 40)),
        ),
        stats={k: dict(mean=[0.0] * 5, scale=[1.0] * 5) for k in ["node", "graph"]},
        selected={k: list(range(5)) for k in ["node", "graph"]},
    )
    write(d / "fold_0.json", cfg)
    write(p / "manifest.json", dict(source_hashes=source_hashes(), smoke=True))
    return p


class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def test_task_graph(self):
        ts = tasks()
        ids = {t["id"] for t in ts}
        self.assertEqual(len(ts), 4000)
        self.assertEqual(len(ids), 4000)
        from collections import Counter

        self.assertEqual(
            Counter((t["kind"] for t in ts)), dict(trial=2360, final=1605, svm=35)
        )
        for t in ts:
            self.assertTrue(set(t["deps"]) <= ids)

    def test_full_self_parameters_and_edge_invariance(self):
        import torch
        from torch_geometric.data import Data
        from learning import build, seed_all

        data = Data(
            x=torch.randn(12, 5),
            edge_index=torch.tensor([[0, 1, 2, 6, 7], [1, 2, 0, 7, 8]]),
            batch=torch.tensor([0] * 6 + [1] * 6),
            graph_features=torch.randn(2, 5),
        )
        for arch in ARCH_SYN:
            spec = dict(arch=arch, edge="full", regime="enhanced")
            seed_all(31)
            a = build(spec, 5, 5, 2, CONFIGS[0])
            seed_all(31)
            b = build(dict(spec, edge="self"), 5, 5, 2, CONFIGS[0])
            self.assertEqual(
                sum((p.numel() for p in a.parameters())),
                sum((p.numel() for p in b.parameters())),
            )
            self.assertTrue(
                all(
                    (
                        torch.equal(v, b.state_dict()[k])
                        for k, v in a.state_dict().items()
                    )
                )
            )
            b.eval()
            first = b(data)[0]
            changed = data.clone()
            changed.edge_index = torch.flip(data.edge_index, [0])
            changed.edge_index = torch.cat([changed.edge_index, changed.edge_index], 1)
            self.assertTrue(torch.equal(first, b(changed)[0]), arch)

    def test_training_resume_and_test_guard(self):
        from learning import fit, Graphs

        model = dict(arch="SAGE", edge="self", regime="enhanced")
        task = dict(
            id="resume_trial",
            dataset="synthetic",
            fold=0,
            model=model,
            kind="trial",
            config=0,
            deps=[],
        )
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            pa = fixture(a)
            pb = fixture(b)
            ds = Graphs(pa, "synthetic", 0, "enhanced")
            with self.assertRaises(RuntimeError):
                ds[32]
            with contextlib.redirect_stdout(io.StringIO()):
                fit(task, pa, "cpu", True)
            os.environ["CONTROLS_TEST_INTERRUPT"] = "0"
            try:
                with self.assertRaises(InterruptedError), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    fit(task, pb, "cpu", True)
            finally:
                os.environ.pop("CONTROLS_TEST_INTERRUPT")
            with contextlib.redirect_stdout(io.StringIO()):
                fit(task, pb, "cpu", True)
            ma = read(pa / "tasks/resume_trial/metrics.json")
            mb = read(pb / "tasks/resume_trial/metrics.json")
            self.assertEqual(
                ma["checkpoint_state_sha256"], mb["checkpoint_state_sha256"]
            )
            self.assertEqual(ma["validation"], mb["validation"])
            self.assertFalse((pa / "tasks/resume_trial/test.npz").exists())

    def test_all_architecture_calibration_and_graph_mlp(self):
        from learning import Graphs, build, calibrate, setup

        with tempfile.TemporaryDirectory() as d:
            p = fixture(d)
            ds = Graphs(p, "synthetic", 0, "enhanced")
            device = setup("cpu")
            for arch in [*ARCH_SYN, "DeepSets", "GraphMLP"]:
                model = build(
                    dict(arch=arch, edge="self" if arch in ARCH_SYN else "none"),
                    5,
                    5,
                    2,
                    CONFIGS[0],
                )
                m, a = calibrate(model, ds, device, 16)
                self.assertTrue(a["training_state_unchanged"])
                self.assertEqual(len(a["training_ids"]), 24)
                if arch == "GraphMLP":
                    self.assertTrue(
                        all((l["weighting"] == "graph" for l in a["layers"]))
                    )


def fake_worker(run):
    import work_queue

    class Process:
        def __init__(self, cmd, **kwargs):
            self.run = Path(cmd[cmd.index("--run") + 1])
            self.id = cmd[cmd.index("--id") + 1]

        def wait(self):
            task = next(
                (t for t in read(self.run / "tasks.json") if t["id"] == self.id)
            )
            for d in task["deps"]:
                if not (self.run / "tasks" / d / "complete.json").exists():
                    raise AssertionError("Dependency started early")
            time.sleep(random.random() / 20)
            with open(self.run / "events.txt", "a") as f:
                f.write(self.id + "\n")
            folder = self.run / "tasks" / self.id
            write(folder / "result.json", {"ok": True})
            commit(folder, ["result.json"])
            return 0

        def poll(self):
            return None

        def send_signal(self, s):
            pass

    work_queue.subprocess.Popen = Process
    work_queue.preflight = lambda device: None
    with contextlib.redirect_stdout(io.StringIO()):
        rc = work_queue.work(run, "cpu", smoke=True, poll_seconds=0.01)
    if rc:
        raise RuntimeError(rc)


class SchedulerTests(unittest.TestCase):
    def test_ten_workers_and_late_arrival(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            ts = [dict(id=f"t{i}", kind="trial", deps=[]) for i in range(20)] + [
                dict(id=f"f{i}", kind="final", deps=[f"t{2 * i}", f"t{2 * i + 1}"])
                for i in range(10)
            ]
            write(p / "tasks.json", ts)
            write(
                p / "manifest.json",
                dict(
                    tasks_sha256=sha(p / "tasks.json"),
                    source_hashes=source_hashes(),
                    smoke=True,
                ),
            )
            ctx = multiprocessing.get_context("fork")
            workers = [ctx.Process(target=fake_worker, args=(d,)) for _ in range(10)]
            for w in workers:
                w.start()
            for w in workers:
                w.join(45)
                self.assertEqual(w.exitcode, 0)
            events = (p / "events.txt").read_text().splitlines()
            self.assertEqual(len(events), 30)
            self.assertEqual(len(set(events)), 30)
            fake_worker(d)
            self.assertEqual(events, (p / "events.txt").read_text().splitlines())

    def test_lock_retained_by_surviving_child(self):
        import subprocess
        from core import lock

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "task.lock"
            script = "import fcntl,os,subprocess,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.8)'],pass_fds=(f.fileno(),),stdout=subprocess.DEVNULL); os._exit(0)"
            parent = subprocess.Popen([sys.executable, "-c", script, str(path)])
            parent.wait(timeout=5)
            with lock(path) as other:
                self.assertIsNone(other)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with lock(path) as other:
                    if other is not None:
                        break
                time.sleep(0.02)
            else:
                self.fail("Surviving child did not release task lock")


if __name__ == "__main__":
    unittest.main(verbosity=2)
