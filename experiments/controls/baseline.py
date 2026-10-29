import io
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
import joblib
from core import read, write, atomic, commit, sha, data_path


def run_svm(task, run, smoke=False):
    folder = Path(run) / "tasks" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    ds = task["dataset"]
    recs = read(Path(run) / "data" / ds / "records.json")
    cfg = read(Path(run) / "data" / ds / f"fold_{task['fold']}.json")
    split = cfg["split"]

    def extract(indices):
        rows = []
        y = []
        for i in indices:
            if sha(data_path(run, recs[i])) != recs[i]["sha256"]:
                raise RuntimeError("Input bytes changed")
            with np.load(data_path(run, recs[i]), allow_pickle=False) as a:
                topo = a["x_topology"].mean(0)[cfg["selected"]["node"]]
                gf = a["graph_features"][cfg["selected"]["graph"]]
                if ds == "synthetic":
                    row = np.r_[topo, gf]
                elif task["regime"] == "native":
                    row = a["x_native"].mean(0)
                else:
                    row = np.r_[a["x_native"].mean(0), topo, gf]
            rows.append(row)
            y.append(recs[i]["label"])
        return (np.array(rows), np.array(y))

    x, y = extract(split["train"])
    v, vy = extract(split["validation"])
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    v = scaler.transform(v)
    grid = [
        (c, g)
        for c in ([1.0] if smoke else [0.01, 0.1, 1.0, 10.0, 100.0])
        for g in ([1.0] if smoke else [0.01, 0.1, 1.0, 10.0, 100.0])
    ]
    history = []
    best = None
    winner = None
    best_score = -1
    for c, g in grid:
        model = SVC(
            C=c,
            gamma=g / x.shape[1],
            kernel="rbf",
            probability=False,
            cache_size=1024,
            random_state=0,
        ).fit(x, y)
        score = balanced_accuracy_score(vy, model.predict(v))
        history.append(
            dict(C=c, gamma_multiplier=g, validation_balanced_accuracy=float(score))
        )
        if score > best_score:
            best_score = score
            best = model
            winner = dict(C=c, gamma_multiplier=g)
    write(
        folder / "selection_frozen.json",
        dict(
            params=winner,
            training_indices=split["train"],
            validation_indices=split["validation"],
        ),
    )
    names = ["metrics.json", "history.json", "best.joblib", "selection_frozen.json"]
    metrics = {}
    for name in ["validation", "test"]:
        xx, yy = extract(split[name])
        xx = scaler.transform(xx)
        pred = best.predict(xx)
        score = best.decision_function(xx)
        met = dict(
            accuracy=float(accuracy_score(yy, pred)),
            balanced_accuracy=float(balanced_accuracy_score(yy, pred)),
            macro_f1=float(f1_score(yy, pred, average="macro", zero_division=0)),
        )
        if score.ndim == 1:
            met["roc_auc"] = float(roc_auc_score(yy, score))
        metrics[name] = met
        np.savez_compressed(
            folder / (name + ".npz"),
            indices=np.array(split[name]),
            labels=yy,
            predicted=pred,
            decision=score,
        )
        names.append(name + ".npz")
    b = io.BytesIO()
    joblib.dump(dict(model=best, scaler=scaler), b)
    atomic(folder / "best.joblib", b.getvalue())
    write(folder / "history.json", history)
    write(
        folder / "metrics.json",
        dict(task=task, params=winner, **metrics, seconds=time.monotonic() - start),
    )
    commit(folder, names)
