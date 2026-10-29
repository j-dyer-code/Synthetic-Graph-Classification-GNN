import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import numpy as np
import importlib.util
from support import load_protocol as load_benchmark_protocol, package_identity
from normalization import calibrated_copy

PACKAGE = Path(__file__).resolve().parent
from lib.common import (
    atomic_bytes,
    atomic_json,
    atomic_npz,
    check_original_source,
    classification_metrics,
    digest,
    exclusive_lock,
    file_hash,
    immutable_json,
    load_ablation,
    stable_seed,
    validate_dataset,
    verify_preparation,
)


class SoftwareInterruption(Exception):
    pass


def setup_torch(source, threads, device):
    global torch, F, Data, DataLoader, models
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from lib import models

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, content):
    buffer = io.BytesIO()
    torch.save(content, buffer)
    atomic_bytes(path, buffer.getvalue())


def load_checkpoint(path, fingerprint):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint identity mismatch")
    return state


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AccessGuard:
    def __init__(self, data, audit, split):
        self.data = data
        self.graphs = data.graphs
        self.expected = {
            item["id"]: item["sha256"] for item in audit["identity"]["graph_files"]
        }
        self.test = set(split["test"])
        self.allow_test = False
        self.accessed = set()
        self.test_reads_before_selection = 0
        self.verified = set()

    def raw(self, index):
        if index in self.test and (not self.allow_test):
            self.test_reads_before_selection += 1
            raise RuntimeError(
                "Test graph arrays cannot be decoded before final checkpoint selection"
            )
        if index not in self.verified:
            if (
                file_hash(self.data.paths[index])
                != self.expected[str(self.graphs[index]["id"])]
            ):
                raise ValueError("Prepared graph byte hash changed")
            self.verified.add(index)
        self.accessed.add(index)
        return self.data.raw(index)

    def audit(self):
        return dict(
            test_reads_before_selection=self.test_reads_before_selection,
            decoded_graphs=len(self.accessed),
            decoded_test_graphs=len(self.accessed & self.test),
            hash_checked_graphs=len(self.verified),
            test_access_enabled_after_selection=self.allow_test,
        )


class SelectedDataset:
    def __init__(self, data, stats, selected):
        self.data, self.stats, self.selected = (data, stats, selected)

    def __len__(self):
        return len(self.data.graphs)

    def __getitem__(self, index):
        raw = self.data.raw(index)

        def scale(values, level):
            stats = self.stats[level]
            values = (
                np.asarray(values, dtype=np.float64) - np.asarray(stats["mean"])
            ) / np.asarray(stats["scale"])
            return np.asarray(
                values[..., self.selected[level]["selected_indices"]], dtype=np.float32
            )

        x, gf = (
            scale(raw["x_topology"], "node"),
            scale(raw["graph_features"], "graph"),
        )
        if not np.isfinite(x).all() or not np.isfinite(gf).all():
            raise FloatingPointError("Nonfinite standardized descriptors")
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.as_tensor(raw["edge_index"], dtype=torch.long),
            y=torch.tensor([self.data.graphs[index]["label"]], dtype=torch.long),
            graph_features=torch.from_numpy(gf.reshape(1, -1)),
            graph_idx=torch.tensor([index]),
            num_nodes=len(x),
        )


def loader(dataset, indices, batch_size, ablation, shuffle_seed=None):
    generator = torch.Generator().manual_seed(
        0 if shuffle_seed is None else shuffle_seed
    )
    return DataLoader(
        dataset,
        batch_sampler=ablation.epoch_batches(indices, batch_size, shuffle_seed),
        num_workers=0,
        generator=generator,
    )


def calibrated_evaluation_copy(model, dataset, split, protocol, ablation, device):
    expected = tuple(split["train"])
    if len(set(expected)) != len(expected):
        raise ValueError("Training calibration partition contains duplicate IDs")
    expected_ids = [str(dataset.data.graphs[i]["id"]) for i in expected]
    observed_passes = []

    def training_batches():
        observed = []
        batches = 0
        for batch in loader(
            dataset, expected, protocol["training"]["batch_size"], ablation
        ):
            actual = batch.graph_idx.cpu().tolist()
            begin = len(observed)
            if (
                actual != list(expected[begin : begin + len(actual)])
                or len(actual) != batch.num_graphs
            ):
                raise ValueError(
                    "Calibration loader emitted a non-training, duplicate or reordered graph ID"
                )
            observed.extend(actual)
            batches += 1
            yield batch
        if tuple(observed) != expected:
            raise ValueError(
                "Calibration loader did not visit the complete training partition"
            )
        ids = [str(dataset.data.graphs[i]["id"]) for i in observed]
        observed_passes.append(
            dict(
                training_graphs=len(observed),
                batches=batches,
                ordered_training_ids_sha256=digest(ids),
            )
        )

    candidate, audit = calibrated_copy(model, training_batches, device)
    if len(observed_passes) != 3 or any(
        (
            p["ordered_training_ids_sha256"] != digest(expected_ids)
            for p in observed_passes
        )
    ):
        raise ValueError(
            "Calibration did not complete three identical training-only passes"
        )
    if any(
        (
            p["training_graphs"] != layer["training_graphs"]
            or p["batches"] != layer["batches"]
            for p, layer in zip(observed_passes, audit["layers"])
        )
    ):
        raise ValueError(
            "Calibration observations disagree with validated loader membership"
        )
    audit["training_graph_ids"] = expected_ids
    audit["training_partition_sha256"] = digest(expected_ids)
    audit["loader_access_validation"] = observed_passes
    return (candidate, audit)


def evaluation(model, batches, device, keep=False):
    model.eval()
    total = correct = count = 0
    ys = []
    scores = []
    embeddings = []
    indices = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            logits, embedding = model(batch)
            loss = F.cross_entropy(logits, batch.y, reduction="sum")
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation")
            total += float(loss)
            count += batch.num_graphs
            correct += int((logits.argmax(1) == batch.y).sum())
            if keep:
                ys.append(batch.y.cpu().numpy())
                scores.append(logits.cpu().numpy())
                embeddings.append(embedding.cpu().numpy())
                indices.extend(batch.graph_idx.cpu().tolist())
    result = dict(loss=total / count, accuracy=correct / count)
    if keep:
        result.update(
            labels=np.concatenate(ys),
            decisions=np.concatenate(scores),
            embeddings=np.concatenate(embeddings),
            indices=indices,
        )
        shifted = result["decisions"].astype(np.float64)
        shifted -= shifted.max(1, keepdims=True)
        exponent = np.exp(shifted)
        result["probabilities"] = exponent / exponent.sum(1, keepdims=True)
        result["predicted"] = result["decisions"].argmax(1)
    return result


def train_epoch(model, batches, optimizer, device, clip):
    model.train()
    total = correct = count = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip, error_if_nonfinite=True
        )
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        correct += int((logits.detach().argmax(1) == batch.y).sum())
        count += batch.num_graphs
    return (total / count, correct / count)


def learner(arch, params, selected, seed, device, protocol, state=None):
    seed_all(seed)
    model = getattr(models, "GraphFamilyClassifier" + arch)(
        len(selected["node"]["selected_indices"]),
        params["hidden_channels"],
        5,
        len(selected["graph"]["selected_indices"]),
        params["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for parameter, slots in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in slots and slots[key].device != parameter.device:
                    raise RuntimeError("Optimizer resume device mismatch")
        restore_rng(state["rng"])
    return (model, optimizer, scheduler)


def study_bytes(study):
    return pickle.dumps(study, protocol=pickle.HIGHEST_PROTOCOL)


def trial_records(study):
    return [
        dict(
            number=t.number,
            state=t.state.name,
            params=t.params,
            history=t.user_attrs.get("history", []),
            objective=t.value,
            error=t.user_attrs.get("error"),
            seconds=t.user_attrs.get("seconds", 0.0),
        )
        for t in study.trials
    ]


def tune(
    folder,
    fingerprint,
    arch,
    selected,
    dataset,
    split,
    protocol,
    seeds,
    ablation,
    device,
    interrupt=None,
):
    import optuna

    study_path = folder / "study.pkl"
    state_path = folder / "trial_state.pt"
    if study_path.exists():
        saved = pickle.loads(study_path.read_bytes())
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Study identity mismatch")
        study = saved["study"]
        search_accumulated = saved.get("search_wall_seconds", 0.0)
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=seeds["tpe"],
                n_startup_trials=0 if protocol["purpose"] == "software_test" else 10,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=protocol["tuning"]["pruner_warmup_steps"]
            ),
        )
        search_accumulated = 0.0
        atomic_bytes(
            study_path,
            pickle.dumps(
                dict(fingerprint=fingerprint, study=study, search_wall_seconds=0.0)
            ),
        )
    active = load_checkpoint(state_path, fingerprint) if state_path.exists() else None
    if (
        active
        and active["trial_number"] < len(study.trials)
        and study.trials[active["trial_number"]].state.is_finished()
    ):
        active = None
        state_path.unlink()
    if active:
        study = pickle.loads(active["study"])
        search_accumulated = active.get("search_wall_seconds", search_accumulated)
    search_started = time.perf_counter()
    while (
        sum((t.state.is_finished() for t in study.trials))
        < protocol["tuning"]["trials"]
    ):
        if active:
            number = active["trial_number"]
            trial = optuna.trial.Trial(study, study.trials[number]._trial_id)
            params = dict(trial.params)
        else:
            running = [t for t in study.trials if not t.state.is_finished()]
            if running:
                raise ValueError(
                    "Study has a running trial without an atomic training checkpoint"
                )
            trial = study.ask()
            number = trial.number
            params = dict(
                hidden_channels=trial.suggest_categorical(
                    "hidden_channels", protocol["tuning"]["hidden_channels"]
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate", **protocol["tuning"]["learning_rate"]
                ),
                dropout=trial.suggest_float("dropout", **protocol["tuning"]["dropout"]),
            )
        trial_seed = seeds["trial"][number]
        try:
            model, optimizer, scheduler = learner(
                arch, params, selected, trial_seed, device, protocol, active
            )
            history = active["history"] if active else []
            seconds = active["seconds"] if active else 0.0
            if not active:
                active = dict(
                    fingerprint=fingerprint,
                    trial_number=number,
                    epoch=0,
                    history=[],
                    seconds=0.0,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng=rng_state(),
                    study=study_bytes(study),
                    search_wall_seconds=search_accumulated
                    + time.perf_counter()
                    - search_started,
                )
                atomic_checkpoint(state_path, active)
            for epoch in (
                []
                if active.get("pruned", False)
                else range(active["epoch"] + 1, protocol["tuning"]["epochs"] + 1)
            ):
                synchronize(device)
                start = time.perf_counter()
                train_loss, train_accuracy = train_epoch(
                    model,
                    loader(
                        dataset,
                        split["train"],
                        protocol["training"]["batch_size"],
                        ablation,
                        stable_seed(seeds["master"], "trial_loader", number, epoch),
                    ),
                    optimizer,
                    device,
                    protocol["training"]["tune_gradient_clip"],
                )
                synchronize(device)
                train_seconds = time.perf_counter() - start
                eval_model, calibration = calibrated_evaluation_copy(
                    model, dataset, split, protocol, ablation, device
                )
                calibration_path = (
                    folder / "calibration" / f"trial_{number}" / f"epoch_{epoch}.json"
                )
                atomic_json(calibration_path, calibration)
                validation_start = time.perf_counter()
                val = evaluation(
                    eval_model,
                    loader(
                        dataset,
                        split["validation"],
                        protocol["training"]["batch_size"],
                        ablation,
                    ),
                    device,
                )
                synchronize(device)
                validation_seconds = time.perf_counter() - validation_start
                del eval_model
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val["loss"])
                synchronize(device)
                seconds += time.perf_counter() - start
                history.append(
                    dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_accuracy=train_accuracy,
                        validation_loss=val["loss"],
                        validation_accuracy=val["accuracy"],
                        learning_rate=lr,
                        training_seconds=train_seconds,
                        validation_seconds=validation_seconds,
                        calibration_seconds=calibration["elapsed_seconds"],
                        calibration_path=str(calibration_path.relative_to(folder)),
                        calibration_sha256=file_hash(calibration_path),
                    )
                )
                trial.report(val["loss"], epoch - 1)
                trial.set_user_attr("history", history)
                trial.set_user_attr("seconds", seconds)
                active = dict(
                    fingerprint=fingerprint,
                    trial_number=number,
                    epoch=epoch,
                    history=history,
                    seconds=seconds,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng=rng_state(),
                    study=study_bytes(study),
                    pruned=trial.should_prune(),
                    search_wall_seconds=search_accumulated
                    + time.perf_counter()
                    - search_started,
                )
                atomic_checkpoint(state_path, active)
                if interrupt == f"trial:{number}:{epoch}":
                    raise SoftwareInterruption(interrupt)
                if trial.should_prune():
                    break
            state = (
                optuna.trial.TrialState.PRUNED
                if active.get("pruned", False)
                else optuna.trial.TrialState.COMPLETE
            )
            study.tell(
                trial,
                history[-1]["validation_loss"]
                if state == optuna.trial.TrialState.COMPLETE
                else None,
                state=state,
            )
        except (FloatingPointError, torch.OutOfMemoryError) as exc:
            trial.set_user_attr(
                "error", dict(type=type(exc).__name__, message=str(exc))
            )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if "eval_model" in locals():
                del eval_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_bytes(
            study_path,
            pickle.dumps(
                dict(
                    fingerprint=fingerprint,
                    study=study,
                    search_wall_seconds=search_accumulated
                    + time.perf_counter()
                    - search_started,
                )
            ),
        )
        state_path.unlink(missing_ok=True)
        active = None
        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        if "scheduler" in locals():
            del scheduler
        print(f"{arch} trial {number}: {study.trials[number].state.name}", flush=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            "No completed finite tuning trial; no final model was selected"
        )
    if (folder / "tuning.json").exists():
        report = json.loads((folder / "tuning.json").read_text())
        if (
            report["trials"] != trial_records(study)
            or report["best_trial"] != study.best_trial.number
        ):
            raise ValueError("Tuning report differs from the committed completed study")
        return (dict(study.best_params), report)
    report = dict(
        tuned=True,
        trials=trial_records(study),
        best_trial=study.best_trial.number,
        objective=protocol["tuning"]["objective"],
        tuning_seconds=search_accumulated + time.perf_counter() - search_started,
        tuning_epoch_seconds=sum(
            (t.user_attrs.get("seconds", 0.0) for t in study.trials)
        ),
        timing_scope="Active search wall time, including suggestions, model initialization, epochs and intermediate checkpoint I/O; excludes interruption downtime and final commit serialization",
    )
    atomic_json(folder / "tuning.json", report)
    return (dict(study.best_params), report)


def final_fit(
    folder,
    fingerprint,
    arch,
    params,
    selected,
    dataset,
    split,
    protocol,
    seeds,
    ablation,
    device,
    interrupt=None,
):
    path = folder / "final_state.pt"
    state = load_checkpoint(path, fingerprint) if path.exists() else None
    model, optimizer, scheduler = learner(
        arch, params, selected, seeds["final"], device, protocol, state
    )
    history = state["history"] if state else []
    best_loss = state["best_loss"] if state else float("inf")
    best_epoch = state["best_epoch"] if state else 0
    no_improve = state["no_improve"] if state else 0
    best = state["best_model"] if state else None
    seconds = state["seconds"] if state else 0.0
    for epoch in range(len(history) + 1, protocol["training"]["max_epochs"] + 1):
        if no_improve >= protocol["training"]["patience"]:
            break
        synchronize(device)
        start = time.perf_counter()
        loss, accuracy = train_epoch(
            model,
            loader(
                dataset,
                split["train"],
                protocol["training"]["batch_size"],
                ablation,
                stable_seed(seeds["master"], "final_loader", epoch),
            ),
            optimizer,
            device,
            protocol["training"]["final_gradient_clip"],
        )
        synchronize(device)
        train_seconds = time.perf_counter() - start
        eval_model, calibration = calibrated_evaluation_copy(
            model, dataset, split, protocol, ablation, device
        )
        calibration_path = folder / "calibration" / "final" / f"epoch_{epoch}.json"
        atomic_json(calibration_path, calibration)
        validation_start = time.perf_counter()
        val = evaluation(
            eval_model,
            loader(
                dataset,
                split["validation"],
                protocol["training"]["batch_size"],
                ablation,
            ),
            device,
        )
        synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["loss"])
        synchronize(device)
        seconds += time.perf_counter() - start
        if val["loss"] < best_loss - protocol["training"]["min_delta"]:
            best_loss = val["loss"]
            best_epoch = epoch
            no_improve = 0
            best = {
                key: value.detach().cpu().clone()
                for key, value in eval_model.state_dict().items()
            }
        else:
            no_improve += 1
        del eval_model
        history.append(
            dict(
                epoch=epoch,
                train_loss=loss,
                train_accuracy=accuracy,
                validation_loss=val["loss"],
                validation_accuracy=val["accuracy"],
                learning_rate=lr,
                training_seconds=train_seconds,
                validation_seconds=validation_seconds,
                calibration_seconds=calibration["elapsed_seconds"],
                calibration_path=str(calibration_path.relative_to(folder)),
                calibration_sha256=file_hash(calibration_path),
            )
        )
        atomic_checkpoint(
            path,
            dict(
                fingerprint=fingerprint,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng_state(),
                history=history,
                best_loss=best_loss,
                best_epoch=best_epoch,
                no_improve=no_improve,
                best_model=best,
                seconds=seconds,
            ),
        )
        if interrupt == f"final:{epoch}":
            raise SoftwareInterruption(interrupt)
    model.load_state_dict(best)
    atomic_checkpoint(folder / "best.pt", best)
    atomic_json(folder / "history.json", history)
    return (model, history, best_epoch, best_loss, seconds)


def write_predictions(path, ids, labels, predicted, probs, decisions):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        ["graph_id", "true_label", "predicted_label", "probability_argmax"]
        + [f"probability_{j}" for j in range(5)]
        + [f"decision_{j}" for j in range(5)]
    )
    for i, graph_id in enumerate(ids):
        writer.writerow(
            [graph_id, int(labels[i]), int(predicted[i]), int(np.argmax(probs[i]))]
            + [repr(float(v)) for v in probs[i]]
            + [repr(float(v)) for v in decisions[i]]
        )
    atomic_bytes(path, stream.getvalue().encode())


def verify_run(folder, fingerprint=None):
    marker = json.loads((folder / "complete.json").read_text())
    config = json.loads((folder / "run_config.json").read_text())
    if marker["fingerprint"] != digest(config) or (
        fingerprint and marker["fingerprint"] != fingerprint
    ):
        raise ValueError("Completed main run identity mismatch")
    for name, expected in marker["files"].items():
        path = folder / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or (not path.resolve().is_relative_to(folder.resolve()))
        ):
            raise ValueError("Unsafe result path")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Completed main artifact changed: " + name)
    return marker


def run(args):
    if args.output.resolve().is_relative_to(PACKAGE.resolve()):
        raise ValueError("Output must be outside the immutable experiment package")
    suite = load_benchmark_protocol(software_test=args.software_test)
    protocol = suite["main"]
    if (
        args.seed not in protocol["seeds"]
        or args.architecture not in protocol["architectures"]
    ):
        raise ValueError("Seed/architecture outside fixed protocol")
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    if args.interrupt_after and (not args.software_test):
        raise ValueError("Intentional interruption is restricted to software fixtures")
    prepared_path = args.prepared or args.output / "prepared"
    original = check_original_source()
    ablation = load_ablation()
    manifest = args.dataset / "manifest.json" if args.dataset.is_dir() else args.dataset
    prepared, prep_identity, audit = verify_preparation(
        prepared_path, manifest, protocol
    )
    data = ablation.PreparedDataset(manifest, cache_graphs=-1)
    validate_dataset(
        data, audit, protocol, new_corpus=prep_identity["dataset_scope"] == "new_corpus"
    )
    expected = audit["identity"]["graph_files"]
    if len(expected) != len(data.graphs) or any(
        (
            str(g["id"]) != r["id"] or g["label"] != r["label"]
            for g, r in zip(data.graphs, expected)
        )
    ):
        raise ValueError("Prepared graph ordering/labels differ")
    seed_dir = prepared_path / f"seed_{args.seed}"
    split_record = json.loads((seed_dir / "split.json").read_text())
    selected = json.loads((seed_dir / "selection.json").read_text())
    stats = json.loads((seed_dir / "normalisation.json").read_text())
    index = {str(g["id"]): i for i, g in enumerate(data.graphs)}
    split = {part: [index[v] for v in ids] for part, ids in split_record["ids"].items()}
    flat = [i for part in split.values() for i in part]
    if set(split) != {"train", "validation", "test"} or sorted(flat) != list(
        range(len(data.graphs))
    ):
        raise ValueError("Splits must be disjoint and exhaustive")
    seeds = dict(
        master=args.seed,
        tpe=stable_seed(args.seed, args.architecture, "tpe"),
        trial=[
            stable_seed(args.seed, args.architecture, "trial", n)
            for n in range(protocol["tuning"]["trials"])
        ],
        final=stable_seed(args.seed, args.architecture, "final"),
        rf_node=selected["node"]["forest_seeds"],
        rf_graph=selected["graph"]["forest_seeds"],
        loader_rule="stable_seed(master,trial_loader,trial_number,epoch); stable_seed(master,final_loader,epoch)",
    )
    device = setup_torch(PACKAGE, args.threads, args.device)
    software = {
        p: importlib.metadata.version(p)
        for p in (
            "numpy",
            "scipy",
            "scikit-learn",
            "torch",
            "torch-geometric",
            "optuna",
        )
    }
    environment = dict(
        machine=platform.machine(),
        threads=args.threads,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        gpu_capability=list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
        deterministic_algorithms=True,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        tf32=False,
    )
    run_config = dict(
        schema_version=1,
        purpose=protocol["purpose"],
        pipeline="main",
        seed=args.seed,
        architecture=args.architecture,
        prepared_fingerprint=prepared["fingerprint"],
        identity=dict(
            protocol=digest(suite),
            original_source=original,
            execution_source=package_identity(),
            dataset_fingerprint=audit["fingerprint"],
        ),
        split_ids=split_record["ids"],
        selected_features=selected,
        normalisation=stats,
        seeds=seeds,
        protocol=protocol,
        normalization_protocol=suite["normalization"],
        raw_graph_cache=-1,
        software=software,
        python=platform.python_version(),
        execution_environment=environment,
        device=args.device,
    )
    fingerprint = digest(run_config)
    folder = args.output / "runs/main" / f"seed_{args.seed}" / args.architecture
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(folder / ".run.lock"):
        if (folder / "complete.json").exists():
            verify_run(folder, fingerprint)
            print(f"Verified complete: {folder}", flush=True)
            return
        immutable_json(folder / "run_config.json", run_config)
        attempts = (
            json.loads((folder / "attempts.json").read_text())
            if (folder / "attempts.json").exists()
            else []
        )
        attempts.append(
            dict(
                started_utc=datetime.now(timezone.utc).isoformat(),
                host=platform.node(),
                software=software,
                device=args.device,
                environment=environment,
            )
        )
        atomic_json(folder / "attempts.json", attempts)
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            guarded = AccessGuard(data, audit, split)
            dataset = SelectedDataset(guarded, stats, selected)
            params, tuning = tune(
                folder,
                fingerprint,
                args.architecture,
                selected,
                dataset,
                split,
                protocol,
                seeds,
                ablation,
                device,
                args.interrupt_after,
            )
            model, history, best_epoch, best_loss, seconds = final_fit(
                folder,
                fingerprint,
                args.architecture,
                params,
                selected,
                dataset,
                split,
                protocol,
                seeds,
                ablation,
                device,
                args.interrupt_after,
            )
            before_test = guarded.audit()
            if (
                before_test["decoded_test_graphs"]
                or before_test["test_reads_before_selection"]
            ):
                raise AssertionError(
                    "Test partition was accessed before checkpoint selection"
                )
            atomic_json(
                folder / "selection_complete.json",
                dict(
                    fingerprint=fingerprint,
                    best_trial=tuning["best_trial"],
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    selected_checkpoint_sha256=file_hash(folder / "best.pt"),
                    data_access_before_test=before_test,
                ),
            )
            best_audit = folder / history[best_epoch - 1]["calibration_path"]
            atomic_bytes(folder / "best_calibration.json", best_audit.read_bytes())
            val = evaluation(
                model,
                loader(
                    dataset,
                    split["validation"],
                    protocol["training"]["batch_size"],
                    ablation,
                ),
                device,
                keep=True,
            )
            if abs(val["loss"] - best_loss) > max(1e-06, abs(best_loss) * 1e-06):
                raise ValueError(
                    "Selected calibrated checkpoint does not replay validation loss"
                )
            guarded.allow_test = True
            synchronize(device)
            start = time.perf_counter()
            test = evaluation(
                model,
                loader(
                    dataset, split["test"], protocol["training"]["batch_size"], ablation
                ),
                device,
                keep=True,
            )
            synchronize(device)
            test_seconds = time.perf_counter() - start
            for part, result in (("validation", val), ("test", test)):
                if result["indices"] != split[part]:
                    raise ValueError("Prediction ordering differs from fixed split")
                ids = [str(data.graphs[i]["id"]) for i in result["indices"]]
                write_predictions(
                    folder / f"{part}_predictions.csv",
                    ids,
                    result["labels"],
                    result["predicted"],
                    result["probabilities"],
                    result["decisions"],
                )
                atomic_npz(
                    folder
                    / (
                        "embeddings.npz"
                        if part == "test"
                        else "validation_embeddings.npz"
                    ),
                    graph_ids=np.asarray(ids, dtype=str),
                    labels=result["labels"],
                    embeddings=result["embeddings"],
                    logits=result["decisions"],
                    probabilities=result["probabilities"],
                )
            atomic_json(folder / "best_parameters.json", params)
            metrics = dict(
                fingerprint=fingerprint,
                classes=protocol["classes"],
                best_epoch=best_epoch,
                epochs_executed=len(history),
                best_validation_loss=best_loss,
                parameter_count=sum((p.numel() for p in model.parameters())),
                timings=dict(
                    tuning_seconds=tuning["tuning_seconds"],
                    tuning_epoch_seconds=tuning["tuning_epoch_seconds"],
                    final_training_seconds=seconds,
                    test_inference_seconds=test_seconds,
                ),
                validation=classification_metrics(
                    val["labels"],
                    val["predicted"],
                    val["probabilities"],
                    protocol["classes"],
                    val["loss"],
                ),
                test=classification_metrics(
                    test["labels"],
                    test["predicted"],
                    test["probabilities"],
                    protocol["classes"],
                    test["loss"],
                ),
                data_access=guarded.audit(),
                memory=dict(
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None,
                ),
            )
            metrics["timing_scopes"] = dict(
                tuning_seconds="Active search wall time including epochs, calibration, validation and intermediate checkpoint I/O; excludes downtime",
                tuning_epoch_seconds="Trial training, calibration and validation epochs including calibration audit serialization",
                final_training_seconds="Final training, calibration and validation epochs including calibration audit serialization",
                test_inference_seconds="Test inference and probability extraction; excludes output serialization",
            )
            atomic_json(folder / "metrics.json", metrics)
            names = [
                "run_config.json",
                "tuning.json",
                "best_parameters.json",
                "history.json",
                "metrics.json",
                "test_predictions.csv",
                "validation_predictions.csv",
                "best.pt",
                "embeddings.npz",
                "validation_embeddings.npz",
                "best_calibration.json",
                "selection_complete.json",
                "study.pkl",
            ]
            names += sorted(
                (
                    str(p.relative_to(folder))
                    for p in (folder / "calibration").rglob("*.json")
                )
            )
            atomic_json(
                folder / "complete.json",
                dict(
                    schema_version=1,
                    purpose=protocol["purpose"],
                    pipeline="main",
                    status="complete",
                    fingerprint=fingerprint,
                    files={name: file_hash(folder / name) for name in names},
                ),
            )
            verify_run(folder, fingerprint)
            (folder / "failure.json").unlink(missing_ok=True)
        except BaseException as exc:
            atomic_json(
                folder / "failure.json",
                dict(
                    type=type(exc).__name__,
                    message=str(exc),
                    purpose=protocol["purpose"],
                ),
            )
            raise
    print(f"Complete: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Tune and evaluate one graph neural network on a prepared partition."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--software-test", action="store_true")
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Prepared splits and transforms; defaults to OUTPUT/prepared",
    )
    parser.add_argument("--interrupt-after", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        run(args)
    except SoftwareInterruption as exc:
        print(f"Intentional software-test interruption: {exc}", flush=True)
        sys.exit(75)


if __name__ == "__main__":
    main()
