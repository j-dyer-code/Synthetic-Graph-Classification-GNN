import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parent
ARCHITECTURES = ["GCN", "GAT", "GATV2", "SAGE", "GIN", "GTN"]
CONDITIONS = [
    "topology",
    "constant",
    "independent",
    "permuted",
    "mixture_025",
    "mixture_050",
    "mixture_075",
    "community",
]


def arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate one node-attribute ablation with optional graph descriptors."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Prepared splits and transforms; defaults to OUTPUT/prepared",
    )
    parser.add_argument("--seed", required=True, type=int, choices=range(42, 47))
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--graph-mode", required=True, choices=["on", "off"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--software-test", action="store_true")
    parser.add_argument(
        "--interrupt-after",
        type=int,
        help="Software fixtures only: exit 75 after this epoch is committed",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.interrupt_after is not None and (
        not args.software_test or args.interrupt_after < 1
    ):
        parser.error("--interrupt-after requires --software-test and a positive epoch")
    return args


def dependencies(threads):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    global np, torch, F, reference, calibrated_copy, digest, file_hash, atomic_json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from normalization import calibrated_copy
    from lib.common import digest, file_hash, atomic_json
    from lib import data as reference


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                f"Existing run identity differs: {path}; use a new output directory"
            )
    else:
        atomic_json(path, value)


def source_identity():
    from support import package_identity

    return package_identity()


class GuardedData:
    def __init__(self, manifest, cache_graphs=64):
        self.delegate = reference.PreparedDataset(manifest, cache_graphs)
        self.__dict__.update(
            {
                key: getattr(self.delegate, key)
                for key in (
                    "path",
                    "root",
                    "manifest",
                    "graphs",
                    "class_names",
                    "node_names",
                    "graph_names",
                    "labels",
                    "paths",
                )
            }
        )
        self.allowed = set()
        self.phase = "unconfigured"
        self.verified = set()
        self.validated_arrays = set()
        self.decoded_by_phase = {}
        self.test_indices = set()
        self.test_access_before_selection = 0
        self.model_selected = False
        for item in self.graphs:
            if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
                raise ValueError("Every graph must declare its SHA256")
        identity = {
            "manifest_sha256": file_hash(self.path),
            "graph_files": [
                {"id": str(g["id"]), "label": g["label"], "sha256": g["sha256"]}
                for g in self.graphs
            ],
        }
        self.fingerprint = digest(identity)

    def permit(self, phase, indices):
        indices = set(indices)
        if not self.model_selected and indices & self.test_indices:
            raise RuntimeError(
                "Test arrays cannot be enabled before checkpoint selection"
            )
        self.phase, self.allowed = (phase, indices)

    def raw(self, index):
        if index not in self.allowed:
            if index in self.test_indices and (not self.model_selected):
                self.test_access_before_selection += 1
            raise RuntimeError(f"Graph {index} is forbidden during {self.phase}")
        if index not in self.verified:
            actual = file_hash(self.paths[index])
            if actual != self.graphs[index]["sha256"]:
                raise ValueError(f"Graph SHA256 differs: {self.graphs[index]['id']}")
            self.verified.add(index)
        self.decoded_by_phase.setdefault(self.phase, set()).add(index)
        raw = self.delegate.raw(index)
        if index in self.validated_arrays:
            return raw
        x, gf, edges, community = (
            raw[k] for k in ("x_topology", "graph_features", "edge_index", "community")
        )
        if x.ndim != 2 or x.shape[1] != 5 or len(x) < 2 or (not np.isfinite(x).all()):
            raise ValueError("Invalid node features")
        if gf.shape != (5,) or not np.isfinite(gf).all():
            raise ValueError("Invalid graph descriptors")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
            raise ValueError("Invalid edges")
        if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError("Edge endpoint outside graph")
        if (
            community.shape != (len(x),)
            or community.dtype.kind not in "iu"
            or (community < 0).any()
        ):
            raise ValueError("Invalid community membership")
        self.validated_arrays.add(index)
        return raw

    def audit(self):
        return {
            "byte_verified_graphs": len(self.verified),
            "accessed_graph_ids_by_phase": {
                phase: [str(self.graphs[i]["id"]) for i in sorted(ids)]
                for phase, ids in self.decoded_by_phase.items()
            },
            "test_graphs_accessed_before_selection": self.test_access_before_selection,
            "model_selected_before_test_access": self.model_selected,
        }


def configuration(args, data, protocol):
    from lib.common import verify_preparation, validate_dataset

    prepared_path = args.prepared or args.output / "prepared"
    prepared, identity, audit = verify_preparation(
        prepared_path, data.path, protocol["main"]
    )
    validate_dataset(
        data,
        audit,
        protocol["main"],
        new_corpus=identity["dataset_scope"] == "new_corpus",
    )
    settings = dict(protocol["ablation"])
    settings.update(device=args.device, threads=args.threads)
    seed_dir = prepared_path / f"seed_{args.seed}"
    recorded = json.loads((seed_dir / "split.json").read_text())
    lookup = {str(g["id"]): i for i, g in enumerate(data.graphs)}
    split = {
        part: [lookup[v] for v in recorded["ids"][part]]
        for part in ("train", "validation", "test")
    }
    expected = reference.stratified_split(
        data, args.seed, settings["train_fraction"], settings["val_fraction"]
    )
    if split != expected:
        raise ValueError("Prepared split differs from the paired split algorithm")
    flat = sum(split.values(), [])
    if sorted(flat) != list(range(len(data.graphs))):
        raise ValueError("Partitions must be disjoint and cover all graphs")
    stats = json.loads((seed_dir / "normalisation.json").read_text())
    if stats["fit_training_graph_ids"] != recorded["ids"]["train"]:
        raise ValueError("Normalisation was not fitted to this training partition")
    data.test_indices = set(split["test"])
    return (settings, split, stats, prepared["fingerprint"])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, device, class_names, collect=False):
    model.eval()
    losses, labels, logits_parts, embeddings, indices = (0.0, [], [], [], [])
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, embedding = model(batch)
            loss = F.cross_entropy(logits, batch.y, reduction="sum")
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation values")
            losses += float(loss)
            labels.extend(batch.y.cpu().tolist())
            logits_parts.append(logits.detach().cpu())
            indices.extend(batch.graph_idx.cpu().tolist())
            if collect:
                if not torch.isfinite(embedding).all():
                    raise FloatingPointError("Nonfinite embeddings")
                embeddings.append(embedding.detach().cpu().numpy())
    logit_tensor = torch.cat(logits_parts)
    probs = torch.softmax(logit_tensor, dim=1).numpy()
    truth = np.asarray(labels, dtype=np.int64)
    metrics = reference.measure_metrics(truth, probs.argmax(1), len(class_names))
    metrics.update(loss=losses / len(truth), graphs=len(truth))
    for entry, name in zip(metrics["per_class"], class_names):
        entry["class_name"] = name
    arrays = {
        "labels": truth,
        "logits": logit_tensor.numpy(),
        "probabilities": probs,
        "indices": np.asarray(indices, dtype=np.int64),
    }
    if collect:
        arrays["embeddings"] = np.concatenate(embeddings)
    return (metrics, arrays)


def save_predictions(run_dir, partition, values, data):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(
        ["graph_id", "true_label", "predicted_label"]
        + [f"probability_{i}" for i in range(len(data.class_names))]
        + [f"logit_{i}" for i in range(len(data.class_names))]
    )
    ids = [str(data.graphs[i]["id"]) for i in values["indices"]]
    for graph_id, label, probs, logits in zip(
        ids, values["labels"], values["probabilities"], values["logits"]
    ):
        writer.writerow(
            [graph_id, int(label), int(probs.argmax())]
            + probs.tolist()
            + logits.tolist()
        )
    reference.atomic_text(run_dir / f"{partition}_predictions.csv", buffer.getvalue())
    final_path = run_dir / f"{partition}_arrays.npz"
    with final_path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            graph_ids=np.asarray(ids),
            **{k: v for k, v in values.items() if k != "indices"},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(final_path.with_suffix(".npz.tmp"), final_path)


def run(args):
    dependencies(args.threads)
    from support import load_protocol

    protocol = load_protocol(software_test=args.software_test)
    manifest = args.dataset / "manifest.json" if args.dataset.is_dir() else args.dataset
    data = GuardedData(manifest, protocol["ablation"]["cache_graphs"])
    settings, split, stats, reference_sha = configuration(args, data, protocol)
    device = reference.setup_torch(settings, ROOT)
    provenance = reference.hardware_provenance(device, settings)
    provenance["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    protocol_path = ROOT / "protocol.json"
    run_config = {
        "workflow": "ablation",
        "purpose": "software_test" if args.software_test else "research",
        "protocol_sha256": file_hash(protocol_path),
        "package_identity": source_identity(),
        "prepared_fingerprint": reference_sha,
        "seed": args.seed,
        "architecture": args.architecture,
        "condition": args.condition,
        "graph_features": args.graph_mode,
        "settings": settings,
        "dataset_fingerprint": data.fingerprint,
        "dataset_manifest_sha256": file_hash(manifest),
        "normalisation": stats,
        "split_ids": {
            key: [str(data.graphs[i]["id"]) for i in value]
            for key, value in split.items()
        },
        "class_names": data.class_names,
        "node_input_dim": 5,
        "graph_input_dim": 5,
        "graph_off_policy": "five standardized graph descriptor channels set to zero",
        "normalization_protocol": protocol.get(
            "normalization", protocol.get("calibration", {})
        ),
        "selection": "minimum validation cross-entropy of each training-only calibrated evaluation copy",
        "tuning": "none; fixed hyperparameters for every ablation condition",
        "runtime_compatibility": {
            "python": provenance["python"].split()[0],
            "software": provenance["software"],
            "device_type": device.type,
            "cuda_version": provenance["cuda_version"],
        },
    }
    fingerprint = digest(run_config)
    run_dir = (
        args.output.resolve()
        / "runs/ablation"
        / f"seed_{args.seed}"
        / args.architecture
        / f"{args.condition}__graph_{args.graph_mode}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete_path = run_dir / "complete.json"
        if complete_path.exists():
            marker = json.loads(complete_path.read_text())
            if marker["fingerprint"] != fingerprint:
                raise ValueError("Completed run has a different identity")
            for name, expected in marker["files"].items():
                if file_hash(run_dir / name) != expected:
                    raise ValueError(f"Completed output differs: {name}")
            print(f"SKIP ablation {run_dir}", flush=True)
            return 0
        immutable_json(run_dir / "run_config.json", run_config)
        attempts_path = run_dir / "attempts.json"
        attempts = (
            json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        )
        attempts.append({"started": reference.utc_now(), "provenance": provenance})
        atomic_json(attempts_path, attempts)
        try:
            return train(
                args,
                data,
                settings,
                split,
                stats,
                device,
                fingerprint,
                run_dir,
                attempts,
            )
        except Exception as exc:
            atomic_json(
                run_dir / "failure.json",
                {
                    "at": reference.utc_now(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "fingerprint": fingerprint,
                },
            )
            raise


def train(args, data, settings, split, stats, device, fingerprint, run_dir, attempts):
    reference.seed_everything(args.seed)
    model = reference.model_for(args.architecture, settings, len(data.class_names)).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    dataset = reference.ConditionDataset(
        data, stats, args.condition, args.graph_mode == "on", args.seed
    )
    latest_path = run_dir / "latest.pt"
    history, best_loss, best_epoch, no_improve, start_epoch = ([], math.inf, 0, 0, 1)
    best_state, best_calibration = (None, None)
    accumulated = 0.0
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Epoch checkpoint identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for parameter, slots in optimizer.state.items():
            for slot in ("exp_avg", "exp_avg_sq"):
                if slot in slots and slots[slot].device != parameter.device:
                    raise RuntimeError("Optimizer moment device mismatch")
        history, best_loss, best_epoch, no_improve = (
            state[k] for k in ("history", "best_loss", "best_epoch", "no_improve")
        )
        best_state, best_calibration = (state["best_model"], state["best_calibration"])
        start_epoch, accumulated = (state["epoch"] + 1, state["training_seconds"])
        reference.restore_rng(state["rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    session_start = time.perf_counter()

    def calibration_loader():
        data.permit("normalization_calibration", split["train"])
        expected = set(split["train"])
        seen = []
        for batch in reference.loader_for(dataset, split["train"], settings):
            indices = batch.graph_idx.tolist()
            if not set(indices).issubset(expected):
                raise RuntimeError(
                    "Nontraining graph entered normalization calibration"
                )
            seen.extend(indices)
            yield batch
        if seen != split["train"]:
            raise RuntimeError(
                "Calibration must process every training graph once in fixed order"
            )

    for epoch in range(start_epoch, settings["max_epochs"] + 1):
        if no_improve >= settings["patience"]:
            break
        data.permit("training", split["train"])
        model.train()
        sync(device)
        started = time.perf_counter()
        train_sum, train_count = (0.0, 0)
        for batch in reference.loader_for(
            dataset, split["train"], settings, args.seed + epoch
        ):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_sum += float(loss.detach()) * batch.num_graphs
            train_count += batch.num_graphs
        sync(device)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        evaluated, calibration = calibrated_copy(model, calibration_loader, device)
        calibration["training_graph_ids"] = [
            str(data.graphs[i]["id"]) for i in split["train"]
        ]
        calibration["training_partition_sha256"] = digest(
            calibration["training_graph_ids"]
        )
        sync(device)
        calibration_seconds = time.perf_counter() - started
        data.permit("validation", split["validation"])
        started = time.perf_counter()
        validation, _ = evaluate(
            evaluated,
            reference.loader_for(dataset, split["validation"], settings),
            device,
            data.class_names,
        )
        sync(device)
        validation_seconds = time.perf_counter() - started
        improved = validation["loss"] < best_loss - settings["min_delta"]
        if improved:
            best_loss, best_epoch, no_improve = (validation["loss"], epoch, 0)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in evaluated.state_dict().items()
            }
            best_calibration = calibration
        else:
            no_improve += 1
        del evaluated
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_sum / train_count,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "selected": improved,
                "training_seconds": training_seconds,
                "calibration_seconds": calibration_seconds,
                "validation_seconds": validation_seconds,
                "calibration": calibration,
            }
        )
        elapsed = accumulated + time.perf_counter() - session_start
        checkpoint_started = time.perf_counter()
        atomic_torch(
            latest_path,
            {
                "fingerprint": fingerprint,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": reference.rng_state(),
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "best_model": best_state,
                "best_calibration": best_calibration,
                "training_seconds": elapsed,
            },
        )
        atomic_json(
            run_dir / f"epoch_{epoch:03d}_checkpoint_timing.json",
            {
                "epoch": epoch,
                "checkpoint_write_seconds": time.perf_counter() - checkpoint_started,
            },
        )
        atomic_json(
            run_dir / "progress.json",
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "updated": reference.utc_now(),
                "fingerprint": fingerprint,
            },
        )
        print(
            f"ABLATION seed={args.seed} arch={args.architecture} condition={args.condition} graph={args.graph_mode} epoch={epoch} calibrated_val_loss={validation['loss']:.8f}",
            flush=True,
        )
        if args.interrupt_after == epoch:
            print(
                "SOFTWARE TEST: intentional interruption after committed epoch",
                flush=True,
            )
            return 75
    if best_state is None or not history:
        raise RuntimeError("No validation-selected checkpoint")
    training_wall = accumulated + time.perf_counter() - session_start
    model.load_state_dict(best_state)
    model.eval()
    atomic_torch(run_dir / "best.pt", best_state)
    atomic_json(run_dir / "selected_calibration.json", best_calibration)
    data.permit("selected_validation", split["validation"])
    started = time.perf_counter()
    validation, val_arrays = evaluate(
        model,
        reference.loader_for(dataset, split["validation"], settings),
        device,
        data.class_names,
        True,
    )
    sync(device)
    selected_validation_seconds = time.perf_counter() - started
    if not math.isclose(validation["loss"], best_loss, rel_tol=1e-06, abs_tol=1e-07):
        raise RuntimeError(
            "Saved calibrated checkpoint did not reproduce selected validation loss"
        )
    data.model_selected = True
    data.permit("selected_test", split["test"])
    started = time.perf_counter()
    test, test_arrays = evaluate(
        model,
        reference.loader_for(dataset, split["test"], settings),
        device,
        data.class_names,
        True,
    )
    sync(device)
    test_seconds = time.perf_counter() - started
    save_predictions(run_dir, "validation", val_arrays, data)
    save_predictions(run_dir, "test", test_arrays, data)
    atomic_json(run_dir / "history.json", history)
    atomic_json(run_dir / "data_access.json", data.audit())
    metrics = {
        "fingerprint": fingerprint,
        "purpose": "software_test" if args.software_test else "research",
        "class_names": data.class_names,
        "best_epoch": best_epoch,
        "epochs_executed": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "validation": validation,
        "test": test,
        "trainable_parameters": sum(
            (p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "training_validation_checkpoint_seconds": training_wall,
        "timing": {
            "training_seconds": sum((h["training_seconds"] for h in history)),
            "calibration_seconds": sum((h["calibration_seconds"] for h in history)),
            "validation_seconds": sum((h["validation_seconds"] for h in history)),
            "selected_validation_seconds": selected_validation_seconds,
            "test_seconds": test_seconds,
            "scope": "completed epoch training/calibration/validation times; interrupted partial epochs excluded; wall total excludes initial loading and final outputs",
        },
        "memory": {
            "maximum_process_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
            "scope": "current process attempt",
        },
        "test_graphs_accessed_before_selection": data.test_access_before_selection,
        "completed": reference.utc_now(),
    }
    atomic_json(run_dir / "metrics.json", metrics)
    attempts[-1]["completed"] = reference.utc_now()
    atomic_json(run_dir / "attempts.json", attempts)
    (run_dir / "failure.json").unlink(missing_ok=True)
    names = [
        "run_config.json",
        "metrics.json",
        "history.json",
        "data_access.json",
        "selected_calibration.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "validation_arrays.npz",
        "test_arrays.npz",
        "best.pt",
        "attempts.json",
    ] + [p.name for p in sorted(run_dir.glob("epoch_*_checkpoint_timing.json"))]
    atomic_json(
        run_dir / "complete.json",
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "purpose": metrics["purpose"],
            "completed": reference.utc_now(),
            "files": {name: file_hash(run_dir / name) for name in names},
        },
    )
    print(
        f"DONE ablation seed_{args.seed} {args.architecture} {args.condition} graph_{args.graph_mode}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
