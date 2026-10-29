import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lib.common import atomic_bytes, atomic_json, digest, file_hash, verify_complete

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Verify training, prediction collection and checkpoint resumption with small fixtures."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = args.output.resolve()
    data, prepared = (root / "dataset", root / "prepared")
    fixture(data)
    invoke(
        [
            ROOT / "prepare.py",
            "--dataset",
            data,
            "--output",
            prepared,
            "--software-test",
        ],
        root / "logs/prepare.log",
    )
    checks = negative_access_checks()
    common = [
        "--dataset",
        data,
        "--prepared",
        prepared,
        "--seed",
        "42",
        "--software-test",
    ]
    for pipeline, script, extra, interruption in (
        ("main", "run_main.py", [], "trial:0:1"),
        (
            "ablation",
            "run_ablation.py",
            ["--condition", "constant", "--graph-mode", "on"],
            "1",
        ),
    ):
        command = [
            ROOT / script,
            *common,
            "--architecture",
            "GCN",
            "--device",
            args.device,
            *extra,
        ]
        baseline = root / "uninterrupted"
        resumed = root / "resumed"
        invoke([*command, "--output", baseline], root / f"logs/{pipeline}_full.log")
        target = resumed / "runs" / pipeline / "seed_42/GCN"
        if pipeline == "ablation":
            target = target / "constant__graph_on"
        if (
            not (target / "complete.json").exists()
            and (not (target / "latest.pt").exists())
            and (not (target / "trial_state.pt").exists())
        ):
            invoke(
                [*command, "--output", resumed, "--interrupt-after", interruption],
                root / f"logs/{pipeline}_interrupt.log",
                expected=(75,),
            )
        invoke([*command, "--output", resumed], root / f"logs/{pipeline}_resume.log")
        first = baseline / "runs" / pipeline / "seed_42/GCN"
        if pipeline == "ablation":
            first = first / "constant__graph_on"
        checks[pipeline] = compare_runs(first, target, pipeline)
    invoke(
        [ROOT / "run_svm.py", *common, "--output", root / "uninterrupted"],
        root / "logs/svm.log",
    )
    svm = root / "uninterrupted/runs/main/seed_42/SVM"
    verify_complete(svm)
    config = read_json(svm / "run_config.json")
    assert config["protocol"]["svm"]["gamma"] == "scale"
    assert (
        config["normalisation"]["fit_training_graph_ids"]
        == config["split_ids"]["train"]
    )
    checks["fixed_svm"] = {
        "completed": True,
        "training_only_scaling": True,
        "gamma": "scale",
    }
    atomic_json(root / "checks.json", checks)
    print(json.dumps(checks, indent=2))


def read_json(path):
    return json.loads(Path(path).read_text())


def fixture(folder):
    protocol = read_json(ROOT / "protocol.json")["main"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "graphs").mkdir(exist_ok=True)
    if (folder / "manifest.json").exists():
        previous = read_json(folder / "manifest.json")
        if previous.get("purpose") != "software_test" or len(previous["graphs"]) != 50:
            raise ValueError(
                "Existing fixture directory is not the 50-graph software fixture"
            )
        for graph in previous["graphs"]:
            if file_hash(folder / graph["path"]) != graph["sha256"]:
                raise ValueError("Existing artificial fixture changed")
        return
    rng = np.random.default_rng(1505)
    graphs = []
    for index in range(50):
        nodes = 64 + index % 3
        pairs = np.asarray([(j, (j + 1) % nodes) for j in range(nodes)], dtype=np.int64)
        edges = np.unique(np.concatenate([pairs, pairs[:, ::-1]], axis=0), axis=0).T
        relative = f"graphs/fixture_{index}.npz"
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            edge_index=edges,
            x_topology=rng.normal(size=(nodes, 5)).astype(np.float32),
            graph_features=rng.normal(size=5).astype(np.float32),
            community=np.arange(nodes, dtype=np.int64) % 3,
        )
        atomic_bytes(folder / relative, buffer.getvalue())
        graphs.append(
            dict(
                id=f"fixture_{index}",
                label=index // 10,
                path=relative,
                sha256=file_hash(folder / relative),
                num_nodes=nodes,
            )
        )
    manifest = dict(
        schema_version=1,
        status="complete",
        purpose="software_test",
        class_names=protocol["classes"],
        node_feature_names=protocol["data"]["node_features"],
        graph_feature_names=protocol["data"]["graph_features"],
        graphs=graphs,
    )
    manifest["dataset_sha256"] = digest(manifest)
    atomic_json(folder / "manifest.json", manifest)


def invoke(arguments, log, expected=(0,), extra_env=None):
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="0",
    )
    if extra_env:
        environment.update(extra_env)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            [sys.executable, *map(str, arguments)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if result.returncode not in expected:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(
            f"Software check failed (exit {result.returncode}): {log}\n{tail}"
        )
    return result.returncode


def scientific(value):
    if isinstance(value, dict):
        return {
            key: scientific(item)
            for key, item in value.items()
            if not key.endswith("seconds") and key != "calibration_sha256"
        }
    if isinstance(value, list):
        return [scientific(item) for item in value]
    return value


def check_calibration(audit, train_count):
    assert audit["fitting_partition"] == "train"
    assert audit["labels_accessed"] is False
    assert audit["layer_order"] == ["bn1", "bn2", "bn3"]
    assert audit["training_state_unchanged"] is True
    assert audit["training_flags_unchanged"] is True
    assert audit["rng_restored"] is True
    for prefix in ("original_state", "learned_parameters", "noncalibration_state"):
        assert audit[prefix + "_sha256_before"] == audit[prefix + "_sha256_after"]
    assert len(audit["layers"]) == 3
    for layer in audit["layers"]:
        assert layer["training_graphs"] == train_count
        assert layer["fitting_partition"] == "train"
        assert layer["dropout"] == "disabled"
        assert layer["moments"]["nonfinite_count"] == 0


def check_completed(folder, pipeline):
    marker = verify_complete(folder)
    assert marker["purpose"] == "software_test"
    config, metrics = (
        read_json(folder / "run_config.json"),
        read_json(folder / "metrics.json"),
    )
    assert config["purpose"] == "software_test"
    train_count = len(config["split_ids"]["train"])
    assert train_count == 30 or train_count == 35
    if pipeline == "main":
        assert metrics["data_access"]["test_reads_before_selection"] == 0
        selection = read_json(folder / "selection_complete.json")
        assert selection["data_access_before_test"]["decoded_test_graphs"] == 0
        for path in (folder / "calibration").rglob("*.json"):
            report = read_json(path)
            check_calibration(report, train_count)
            assert report["training_graph_ids"] == config["split_ids"]["train"]
        assert read_json(folder / "tuning.json")["trials"]
    else:
        assert metrics["test_graphs_accessed_before_selection"] == 0
        access = read_json(folder / "data_access.json")
        assert access["model_selected_before_test_access"] is True
        assert access["test_graphs_accessed_before_selection"] == 0
        assert set(
            access["accessed_graph_ids_by_phase"]["normalization_calibration"]
        ) == set(config["split_ids"]["train"])
        for row in read_json(folder / "history.json"):
            check_calibration(row["calibration"], train_count)
    assert np.isclose(
        metrics["validation"]["loss"],
        metrics["best_validation_loss"],
        rtol=1e-06,
        atol=1e-07,
    )
    return marker


def compare_runs(reference, resumed, pipeline):
    import torch

    first, second = (
        check_completed(reference, pipeline),
        check_completed(resumed, pipeline),
    )
    assert first["fingerprint"] == second["fingerprint"]
    if scientific(read_json(reference / "history.json")) != scientific(
        read_json(resumed / "history.json")
    ):
        raise AssertionError(f"{pipeline}: resumed scientific epoch history differs")
    metrics_a, metrics_b = (
        read_json(reference / "metrics.json"),
        read_json(resumed / "metrics.json"),
    )
    for key in (
        "best_epoch",
        "epochs_executed",
        "best_validation_loss",
        "validation",
        "test",
    ):
        if metrics_a[key] != metrics_b[key]:
            raise AssertionError(f"{pipeline}: resumed {key} differs")
    for part in ("validation", "test"):
        if (reference / f"{part}_predictions.csv").read_bytes() != (
            resumed / f"{part}_predictions.csv"
        ).read_bytes():
            raise AssertionError(f"{pipeline}: resumed {part} predictions differ")
    weights_a = torch.load(reference / "best.pt", map_location="cpu", weights_only=True)
    weights_b = torch.load(resumed / "best.pt", map_location="cpu", weights_only=True)
    if weights_a.keys() != weights_b.keys() or any(
        (not torch.equal(weights_a[key], weights_b[key]) for key in weights_a)
    ):
        raise AssertionError(
            f"{pipeline}: resumed calibrated checkpoint tensors differ"
        )
    if pipeline == "main":
        tuning_a, tuning_b = (
            read_json(reference / "tuning.json"),
            read_json(resumed / "tuning.json"),
        )
        for key in ("trials", "best_trial", "objective"):
            if scientific(tuning_a[key]) != scientific(tuning_b[key]):
                raise AssertionError(f"main: resumed adaptive tuning {key} differs")
        arrays = ("embeddings.npz", "validation_embeddings.npz")
    else:
        arrays = ("test_arrays.npz", "validation_arrays.npz")
    for name in arrays:
        with np.load(reference / name, allow_pickle=False) as a, np.load(
            resumed / name, allow_pickle=False
        ) as b:
            assert a.files == b.files
            for key in a.files:
                if not np.array_equal(a[key], b[key]):
                    raise AssertionError(f"{pipeline}: resumed {name}/{key} differs")
    return dict(
        exact_scientific_history=True,
        exact_predictions=True,
        exact_embeddings=True,
        exact_calibrated_checkpoint=True,
        test_arrays_blocked_before_selection=True,
        adaptive_tuning_exact=pipeline == "main",
    )


def negative_access_checks():
    import torch
    import run_main
    import run_ablation

    data = SimpleNamespace(
        graphs=[{"id": "forbidden_fixture"}],
        paths=[Path("forbidden-array-must-not-be-opened")],
    )
    audit = {
        "identity": {"graph_files": [{"id": "forbidden_fixture", "sha256": "0" * 64}]}
    }
    guard = run_main.AccessGuard(data, audit, {"test": [0]})
    try:
        guard.raw(0)
    except RuntimeError as error:
        assert "before final checkpoint selection" in str(error)
    else:
        raise AssertionError("Main test-array guard did not reject premature access")
    assert guard.test_reads_before_selection == 1 and (not guard.accessed)
    guard = run_ablation.GuardedData.__new__(run_ablation.GuardedData)
    guard.allowed = set()
    guard.test_indices = {0}
    guard.model_selected = False
    guard.test_access_before_selection = 0
    guard.phase = "software_guard_test"
    for operation in (lambda: guard.permit("forbidden", [0]), lambda: guard.raw(0)):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "Ablation test-array guard did not reject premature access"
            )
    assert guard.test_access_before_selection == 1
    original_loader, original_calibrator = (run_main.loader, run_main.calibrated_copy)
    dataset = SimpleNamespace(
        data=SimpleNamespace(graphs=[{"id": f"fixture_{i}"} for i in range(3)])
    )

    def consume_batches(model, factory, device):
        list(factory())
        return (None, {"layers": []})

    try:
        run_main.calibrated_copy = consume_batches
        for indices in ([2], [1, 0], [0, 0], [0]):
            batch = SimpleNamespace(
                graph_idx=torch.tensor(indices), num_graphs=len(indices)
            )
            run_main.loader = lambda *arguments, batch=batch, **keywords: iter([batch])
            try:
                run_main.calibrated_evaluation_copy(
                    None,
                    dataset,
                    {"train": [0, 1]},
                    {"training": {"batch_size": 2}},
                    None,
                    "cpu",
                )
            except ValueError as error:
                assert "Calibration loader" in str(error)
            else:
                raise AssertionError(
                    "Main calibration accepted held-out, reordered, duplicate or incomplete training IDs"
                )
    finally:
        run_main.loader, run_main.calibrated_copy = (
            original_loader,
            original_calibrator,
        )
    return dict(
        main_premature_test_read_rejected=True,
        ablation_premature_test_read_rejected=True,
        ablation_premature_test_permission_rejected=True,
        calibration_nontraining_reordered_duplicate_and_missing_ids_rejected=True,
    )


if __name__ == "__main__":
    main()
