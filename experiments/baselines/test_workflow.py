import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from common import (
    ROOT,
    atomic_json,
    atomic_npz,
    complete,
    digest,
    file_hash,
    load_protocol,
)


def fixture(root):
    protocol = load_protocol(True)
    cache = root / "graphs"
    cache.mkdir()
    prepared = root / "prepared"
    prepared.mkdir()
    ids = [str(i) for i in range(50)]
    labels = np.repeat(np.arange(5), 10)
    graphs = []
    for index, (gid, label) in enumerate(zip(ids, labels)):
        n = 12 + index % 4
        pairs = {(u, (u + 1) % n) for u in range(n)}
        if index % 2:
            pairs.update(((u, (u + 3) % n) for u in range(n)))
        directed = sorted(pairs | {(v, u) for u, v in pairs})
        path = cache / f"graph_{index}.npz"
        atomic_npz(
            path,
            edge_index=np.asarray(directed, dtype=np.int64).T,
            x_topology=np.zeros((n, 5)),
            graph_features=np.zeros(5),
            community=np.zeros(n, dtype=np.int64),
        )
        graphs.append(
            dict(
                id=gid,
                label=int(label),
                path=path.name,
                sha256=file_hash(path),
                num_nodes=n,
            )
        )
    atomic_json(
        cache / "manifest.json",
        dict(
            schema_version=1,
            status="complete",
            purpose="software_test",
            graphs=graphs,
            class_names=protocol["classes"],
        ),
    )
    raw_identity = dict(
        manifest_sha256=file_hash(cache / "manifest.json"),
        graph_files=[{k: g[k] for k in ("id", "label", "sha256")} for g in graphs],
    )
    identity = dict(purpose="software_test", dataset_fingerprint=digest(raw_identity))
    atomic_json(prepared / "identity.json", identity)
    atomic_json(prepared / "protocol.json", protocol)
    atomic_json(prepared / "dataset_audit.json", dict(identity=raw_identity))
    (prepared / "dataset_descriptors.csv").write_text(
        "graph_id,label\n" + "".join((f"{g},{y}\n" for g, y in zip(ids, labels)))
    )
    files = [
        "identity.json",
        "protocol.json",
        "dataset_audit.json",
        "dataset_descriptors.csv",
    ]
    for seed in protocol["seeds"]:
        split = {p: [] for p in ("train", "validation", "test")}
        rng = np.random.default_rng(seed)
        for label in range(5):
            group = rng.permutation(np.arange(label * 10, (label + 1) * 10)).tolist()
            for part, group_slice in [
                ("train", group[:7]),
                ("validation", group[7:8]),
                ("test", group[8:]),
            ]:
                split[part].extend((str(i) for i in group_slice))
        folder = prepared / f"seed_{seed}"
        folder.mkdir()
        X = rng.normal(size=(50, 6))
        X[:, :5] += 2 * np.eye(5)[labels]
        train = [int(i) for i in split["train"]]
        mean = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        X = (X - mean) / np.where(scale == 0, 1, scale)
        atomic_json(folder / "split.json", dict(ids=split))
        atomic_json(
            folder / "selection.json",
            dict(node_indices=[0, 1, 2], graph_indices=[0, 1, 2]),
        )
        atomic_json(
            folder / "normalisation.json",
            dict(mean=mean.tolist(), scale=scale.tolist()),
        )
        atomic_npz(
            folder / "svm_arrays.npz", graph_ids=np.array(ids), labels=labels, X=X
        )
        files.extend(
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
    complete(prepared, digest(identity), "software_test", files)
    return (cache, prepared)


class Workflow(unittest.TestCase):
    def test_fresh_fixture_resume_and_input_guards(self):
        with tempfile.TemporaryDirectory(prefix="baseline-fixture-") as name:
            root = Path(name)
            cache, prepared = fixture(root)
            output = root / "baselines"
            env = dict(
                os.environ,
                PYTHONHASHSEED="0",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )

            def invoke(args, success=True):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "run.py"), *map(str, args)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                return result

            common = ["--prepared", prepared, "--output", output, "--smoke"]
            invoke(["prepare", *common, "--dataset", cache])
            for seed in range(42, 47):
                invoke(["run", *common, "--seed", seed])
            paths = list((output / "runs").glob("seed_*/*/test_predictions.csv"))
            self.assertEqual(len(paths), 10)
            before = {str(p): file_hash(p) for p in paths}
            invoke(["prepare", *common, "--dataset", cache])
            invoke(["run", *common, "--seed", 42])
            self.assertEqual(before, {str(p): file_hash(p) for p in paths})
            for seed in range(42, 47):
                split = json.loads((prepared / f"seed_{seed}/split.json").read_text())[
                    "ids"
                ]
                for model in ("Graph2Vec_SVM", "Descriptor_SVM_tuned"):
                    import pandas as pd

                    predictions = pd.read_csv(
                        output / f"runs/seed_{seed}/{model}/test_predictions.csv",
                        dtype={"graph_id": str},
                    )
                    self.assertEqual(predictions.graph_id.tolist(), split["test"])
                    self.assertEqual(len(predictions), 10)
            invoke(
                ["run", "--prepared", prepared, "--output", output, "--seed", 42],
                success=False,
            )
            identity_path = output / "identity.json"
            original = identity_path.read_text()
            identity = json.loads(original)
            identity["source"]["classifiers.py"] = "0" * 64
            atomic_json(identity_path, identity)
            invoke(["run", *common, "--seed", 42], success=False)
            identity_path.write_text(original)
            changed = output / "runs/seed_42/Graph2Vec_SVM/test_predictions.csv"
            changed.write_text(changed.read_text() + "\n")
            invoke(["run", *common, "--seed", 42], success=False)
            raw = cache / "graph_0.npz"
            raw.write_bytes(raw.read_bytes() + b"changed")
            invoke(["prepare", *common, "--dataset", cache], success=False)


if __name__ == "__main__":
    unittest.main()
