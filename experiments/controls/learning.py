import functools
import io
import json
import os
import random
import time
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from reference_models import models
from normalization import calibrated_copy, state_hash
from core import read, write, atomic, digest, sha, commit, data_path
from protocol import CONFIGS, SETTINGS


def seed_value(*xs):
    return int(digest(xs)[:8], 16) % (2**31 - 1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda[:index]")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch build"
            )
        torch.cuda.set_device(device)
    return device


def rng_state(device):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_rng(s, device):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"], device)


def save_torch(p, v):
    f = io.BytesIO()
    torch.save(v, f)
    atomic(p, f.getvalue())


class DeepSets(models.GraphFamilyClassifierBase):
    def define_conv_layer_1(self, a, b):
        return IgnoreEdges(a, b)

    def define_conv_layer_2(self, a, b):
        return IgnoreEdges(a, b)


class IgnoreEdges(nn.Linear):
    def forward(self, x, edge_index):
        return super().forward(x)


class GraphMLP(nn.Module):
    def __init__(self, g, h, c, drop):
        super().__init__()
        self.lin1 = nn.Linear(g, h)
        self.lin2 = nn.Linear(h, h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, c)
        self.bn1 = nn.BatchNorm1d(h)
        self.bn2 = nn.BatchNorm1d(h)
        self.bn3 = nn.BatchNorm1d(h)
        self.dropout = nn.Dropout(drop)

    def forward(self, data):
        x = data.graph_features.reshape(-1, self.lin1.in_features)
        x = self.dropout(self.bn1(F.relu(self.lin1(x))))
        x = self.dropout(self.bn2(F.relu(self.lin2(x))))
        x = self.dropout(self.bn3(F.relu(self.fc1(x))))
        return (self.fc2(x), x)


def build(spec, xdim, gdim, classes, params):
    if spec["arch"] == "GraphMLP":
        return GraphMLP(gdim, params["hidden"], classes, params["dropout"])
    cls = (
        DeepSets
        if spec["arch"] == "DeepSets"
        else getattr(models, "GraphFamilyClassifier" + spec["arch"])
    )
    if spec["edge"] == "self":

        class SelfOnly(cls):
            def forward(self, data):
                nodes = torch.arange(data.x.shape[0], device=data.x.device)
                d = types.SimpleNamespace(
                    x=data.x,
                    edge_index=torch.stack([nodes, nodes]),
                    batch=data.batch,
                    graph_features=data.graph_features,
                )
                return super().forward(d)

        cls = SelfOnly
    return cls(xdim, params["hidden"], classes, gdim, params["dropout"])


class Graphs:
    def __init__(self, run, dataset, fold, regime):
        self.run = Path(run).resolve()
        self.records = read(self.run / "data" / dataset / "records.json")
        self.config = read(Path(run) / "data" / dataset / f"fold_{fold}.json")
        self.dataset = dataset
        self.regime = regime
        self.allowed_test = False
        self.test = set(self.config["split"]["test"])
        self.classes = len(set((r["label"] for r in self.records)))

    def __len__(self):
        return len(self.records)

    @functools.lru_cache(maxsize=None)
    def raw(self, i):
        p = data_path(self.run, self.records[i])
        if sha(p) != self.records[i]["sha256"]:
            raise RuntimeError("Input bytes changed: " + str(p))
        with np.load(p, allow_pickle=False) as a:
            return {k: a[k].copy() for k in a.files}

    def __getitem__(self, i):
        i = int(i)
        if i in self.test and (not self.allowed_test):
            raise RuntimeError("Test data access before final model selection")
        r = self.raw(i)
        s = self.config["stats"]
        sel = self.config["selected"]

        def scale(v, k):
            return (
                np.asarray(v, dtype=np.float64) - np.asarray(s[k]["mean"])
            ) / np.asarray(s[k]["scale"])

        topo = scale(r["x_topology"], "node")[:, sel["node"]]
        gf = scale(r["graph_features"], "graph")[sel["graph"]]
        if self.dataset == "synthetic":
            x = topo
        else:
            native = scale(r["x_native"], "native")
            if (
                r["x_native"].shape[1] == 1
                and np.all(r["x_native"] == 1)
                and (s["native"]["mean"] == [1.0])
                and (s["native"]["scale"] == [1.0])
            ):
                native = np.ones_like(native)
            x = np.c_[
                native, topo if self.regime == "enhanced" else np.zeros_like(topo)
            ]
            if self.regime == "native":
                gf = np.zeros_like(gf)
        if not np.isfinite(x).all() or not np.isfinite(gf).all():
            raise ValueError("Nonfinite input")
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(r["edge_index"], dtype=torch.long),
            y=torch.tensor([self.records[i]["label"]], dtype=torch.long),
            graph_features=torch.tensor(gf.reshape(1, -1), dtype=torch.float32),
            graph_idx=torch.tensor([i]),
            num_nodes=len(x),
        )

    def release(self):
        self.raw.cache_clear()


def batches(ds, indices, batch_size, shuffle_seed=None):
    order = list(indices)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        tail = chunks.pop()
        chunks[-1].extend(tail)
    if [i for chunk in chunks for i in chunk] != order:
        raise RuntimeError("Batch construction changed membership or order")
    if any((len(c) < 2 for c in chunks)):
        raise ValueError("At least two graphs required per batch for BatchNorm")
    return DataLoader(
        ds,
        batch_sampler=chunks,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )


def calibrate(model, ds, device, bs):
    expected = list(ds.config["split"]["train"])
    passes = []

    def train_batches():
        observed = []
        for b in batches(ds, expected, bs):
            observed += b.graph_idx.tolist()
            yield b
        if observed != expected:
            raise RuntimeError("Calibration membership mismatch")
        passes.append(digest(observed))

    candidate, audit = calibrated_copy(model, train_batches, device)
    if passes != [digest(expected)] * 3:
        raise RuntimeError("Incomplete training-only calibration")
    audit["training_ids"] = [ds.records[i]["id"] for i in expected]
    if isinstance(model, GraphMLP):
        for layer in audit["layers"]:
            layer["weighting"] = "graph"
    return (candidate, audit)


def evaluate(model, ds, indices, device, bs):
    model.eval()
    ys = []
    logits = []
    ids = []
    emb = []
    with torch.no_grad():
        for b in batches(ds, indices, bs):
            ids += b.graph_idx.tolist()
            ys.append(b.y.numpy())
            b = b.to(device)
            l, e = model(b)
            logits.append(l.cpu().numpy())
            emb.append(e.cpu().numpy())
    if ids != list(indices):
        raise RuntimeError("Evaluation membership or order mismatch")
    y = np.concatenate(ys)
    z = np.concatenate(logits)
    prob = torch.softmax(torch.tensor(z), dim=1).numpy()
    pred = prob.argmax(1)
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite predictions")
    met = dict(
        loss=float(log_loss(y, prob, labels=list(range(ds.classes)))),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        roc_auc=float(
            roc_auc_score(y, prob[:, 1])
            if ds.classes == 2
            else roc_auc_score(y, prob, multi_class="ovr", average="macro")
        ),
    )
    return (
        met,
        dict(
            indices=np.array(ids),
            labels=y,
            logits=z,
            probabilities=prob,
            embeddings=np.concatenate(emb),
        ),
    )


def choose(task, run):
    if task["kind"] == "trial":
        return task["config"]
    results = {
        d: read(Path(run) / "tasks" / d / "metrics.json")
        for d in task["deps"]
        if "/trial_" in d
    }
    scores = {}
    for d, r in results.items():
        scores.setdefault(r["config_index"], []).append(r["validation"]["loss"])
    expected = 2 if task["mode"] == "paired" else 1
    if any((len(v) != expected for v in scores.values())):
        raise RuntimeError("Selection dependencies inconsistent")
    return min(scores, key=lambda c: (float(np.mean(scores[c])), c))


def fit(task, run, device="cuda", smoke=False):
    started = time.monotonic()
    device = setup(device)
    folder = Path(run) / "tasks" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    ci = choose(task, run)
    params = CONFIGS[ci]
    spec = task["model"]
    ds = Graphs(run, task["dataset"], task["fold"], spec["regime"])
    split = ds.config["split"]
    phase = 0 if task["kind"] == "trial" else 1 + task["initialization"]
    seed = seed_value(task["dataset"], task["fold"], spec["arch"], ci, phase)
    seed_all(seed)
    first = ds[split["train"][0]]
    model = build(
        spec, first.x.shape[1], first.graph_features.shape[1], ds.classes, params
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    limit = SETTINGS["tuning_epochs" if task["kind"] == "trial" else "final_epochs"]
    patience = SETTINGS[
        "tuning_patience" if task["kind"] == "trial" else "final_patience"
    ]
    min_epoch = SETTINGS[
        "tuning_min_epochs" if task["kind"] == "trial" else "final_min_epochs"
    ]
    if smoke:
        limit = 2
        min_epoch = 1
        patience = 2
    identity = digest(
        dict(
            task=task,
            params=params,
            data=ds.config,
            source=read(Path(run) / "manifest.json")["source_hashes"],
        )
    )
    start_epoch = 0
    best_loss = float("inf")
    bad = 0
    history = []
    best_state = None
    best_audit = None
    best_epoch = None
    previous_seconds = 0
    resume = folder / "resume.pt"
    if resume.exists():
        s = torch.load(resume, map_location="cpu", weights_only=False)
        if s["identity"] != identity:
            raise RuntimeError("Resume identity differs")
        if s.get("device_type") != device.type or s.get("torch_version") != str(
            torch.__version__
        ):
            raise RuntimeError(
                "Resume requires the same CPU/CUDA backend and PyTorch version"
            )
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        restore_rng(s["rng"], device)
        start_epoch = s["epoch"] + 1
        best_loss = s["best_loss"]
        bad = s["bad"]
        history = s["history"]
        best_state = s["best_state"]
        best_audit = s["best_audit"]
        best_epoch = s["best_epoch"]
        previous_seconds = s["seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, limit):
        if bad >= patience and epoch >= min_epoch:
            break
        model.train()
        train_loss = 0
        count = 0
        for b in batches(
            ds, split["train"], params["batch_size"], seed_value(seed, "epoch", epoch)
        ):
            b = b.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, _ = model(b)
            loss = F.cross_entropy(z, b.y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += float(loss.detach()) * b.num_graphs
            count += b.num_graphs
        candidate, audit = calibrate(model, ds, device, params["batch_size"])
        val, _ = evaluate(
            candidate, ds, split["validation"], device, params["batch_size"]
        )
        if val["loss"] < best_loss - 1e-08:
            best_loss = val["loss"]
            bad = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()
            }
            best_audit = audit
            best_epoch = epoch
        else:
            bad += 1
        history.append(dict(epoch=epoch, train_loss=train_loss / count, validation=val))
        save_torch(
            resume,
            dict(
                identity=identity,
                device_type=device.type,
                torch_version=str(torch.__version__),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(device),
                epoch=epoch,
                best_loss=best_loss,
                bad=bad,
                history=history,
                best_state=best_state,
                best_audit=best_audit,
                best_epoch=best_epoch,
                seconds=previous_seconds + time.monotonic() - started,
            ),
        )
        print(
            json.dumps(dict(task=task["id"], epoch=epoch, validation_loss=val["loss"])),
            flush=True,
        )
        if os.environ.get("CONTROLS_STOP") == "1":
            raise SystemExit(75)
        if os.environ.get("CONTROLS_TEST_INTERRUPT") == str(epoch):
            raise InterruptedError("Requested checkpoint test interruption")
    if best_state is None:
        raise RuntimeError("No selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation, arr = evaluate(
        model, ds, split["validation"], device, params["batch_size"]
    )
    np.savez_compressed(folder / "validation.npz", **arr)
    save_torch(folder / "best.pt", best_state)
    write(folder / "calibration.json", best_audit)
    write(folder / "history.json", history)
    met = dict(
        task=task,
        config_index=ci,
        params=params,
        seed=seed,
        identity=identity,
        selected_epoch=best_epoch,
        epochs_completed=len(history),
        hit_epoch_cap=len(history) == limit,
        validation=validation,
        parameter_count=sum((p.numel() for p in model.parameters())),
        checkpoint_state_sha256=state_hash(model),
        seconds=previous_seconds + time.monotonic() - started,
        gpu_peak_bytes=torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else 0,
        device=str(device),
    )
    names = [
        "metrics.json",
        "best.pt",
        "calibration.json",
        "history.json",
        "validation.npz",
    ]
    if task["kind"] == "final":
        write(
            folder / "selection_frozen.json",
            dict(
                config_index=ci,
                selected_epoch=best_epoch,
                state_sha256=state_hash(model),
                dependencies=task["deps"],
            ),
        )
        ds.allowed_test = True
        test, arr = evaluate(model, ds, split["test"], device, params["batch_size"])
        met["test"] = test
        np.savez_compressed(folder / "test.npz", **arr)
        names += ["test.npz", "selection_frozen.json"]
    write(folder / "metrics.json", met)
    commit(folder, names)
    resume.unlink(missing_ok=True)
    ds.release()
