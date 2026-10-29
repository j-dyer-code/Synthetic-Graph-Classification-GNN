from contextlib import contextmanager
import hashlib
import json
import os
import socket
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def package_identity():
    from support import package_identity as identity

    return identity()


def source_identity():
    return package_identity()


def check_original_source(root=None):
    return {"lib/models.py": file_hash(ROOT / "lib/models.py")}


def load_protocol(path=None, smoke=False):
    from support import load_protocol as load

    return load(software_test=smoke)["main"]


def load_ablation(path=None):
    from . import data

    return data


def validate_dataset(data, audit, protocol, new_corpus=False):
    if list(data.class_names) != protocol["classes"]:
        raise ValueError("Class order differs from protocol")
    if (
        list(data.node_names) != protocol["data"]["node_features"]
        or list(data.graph_names) != protocol["data"]["graph_features"]
    ):
        raise ValueError("Descriptor names/order differ from protocol")
    expected = protocol["data"]["graphs_per_family"]
    if audit["class_counts"] != [expected] * 5 or audit["graphs"] != expected * 5:
        raise ValueError("Graph counts differ from protocol")
    purpose = data.manifest.get("purpose", "research")
    if protocol["purpose"] == "software_test":
        if purpose != "software_test":
            raise ValueError("Software tests require an explicit software_test dataset")
    elif purpose != "research":
        raise ValueError("Research runs require a research dataset")
    elif not new_corpus:
        path = ROOT.parent / "data/published_identity.json"
        published = json.loads(path.read_text())["graphs"]
        if audit["identity"]["graph_files"] != published:
            raise ValueError(
                "Graph contents differ from the published corpus; use --new-corpus during preparation for an independently generated corpus"
            )


def verify_complete(folder, fingerprint=None):
    folder = Path(folder)
    marker = json.loads((folder / "complete.json").read_text())
    if marker.get("purpose") not in {"research", "software_test"} or not marker.get(
        "files"
    ):
        raise ValueError("Invalid completion inventory")
    if fingerprint is not None and marker.get("fingerprint") != fingerprint:
        raise ValueError("Completion fingerprint mismatch")
    for name, expected in marker["files"].items():
        relative = Path(name)
        path = (folder / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or (not path.is_relative_to(folder.resolve()))
        ):
            raise ValueError("Unsafe completion path")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing or modified completed artifact: {name}")
    for identity_name in ("run_config.json", "identity.json"):
        if identity_name in marker["files"]:
            value = json.loads((folder / identity_name).read_text())
            if (
                digest(value) != marker["fingerprint"]
                or value["purpose"] != marker["purpose"]
            ):
                raise ValueError("Completion identity mismatch")
    return marker


def verify_preparation(folder, manifest, protocol):
    folder = Path(folder)
    marker = verify_complete(folder)
    identity = json.loads((folder / "identity.json").read_text())
    audit = json.loads((folder / "dataset_audit.json").read_text())
    required = {
        "identity.json",
        "protocol.json",
        "dataset_audit.json",
        "dataset_descriptors.csv",
    }
    required.update(
        (
            f"seed_{seed}/{name}"
            for seed in protocol["seeds"]
            for name in (
                "split.json",
                "selection.json",
                "normalisation.json",
                "svm_arrays.npz",
            )
        )
    )
    if set(marker["files"]) != required:
        raise ValueError("Preparation completion inventory differs from the protocol")
    if (
        identity["protocol"] != digest(protocol)
        or marker["purpose"] != protocol["purpose"]
        or identity["original_source"] != check_original_source()
        or (identity["preparation_source"] != source_identity())
    ):
        raise ValueError(
            "Preparation source, purpose or protocol differs; prepare a fresh output directory"
        )
    if (
        file_hash(manifest) != audit["identity"]["manifest_sha256"]
        or digest(audit["identity"]) != audit["fingerprint"]
        or identity["dataset_fingerprint"] != audit["fingerprint"]
    ):
        raise ValueError("Prepared dataset identity changed")
    return (marker, identity, audit)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_json(path, value):
    atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(value):
            raise ValueError(
                f"Existing identity differs: {path}; choose a new output directory"
            )
    else:
        atomic_json(path, value)


def atomic_npz(path, **arrays):
    import io

    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    atomic_bytes(path, output.getvalue())


def stable_seed(master, *parts):
    return int.from_bytes(
        hashlib.sha256(canonical([int(master), *parts]).encode()).digest()[:4], "big"
    )


@contextmanager
def exclusive_lock(path):
    import fcntl

    path = Path(path)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Output is locked by an active process: {path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def classification_metrics(labels, predicted, probabilities, classes, loss=None):
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(classes)
    if (
        labels.shape != predicted.shape
        or probabilities.shape != (len(labels), count)
        or (not len(labels))
    ):
        raise ValueError("Prediction dimensions differ")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (not np.allclose(probabilities.sum(1), 1, atol=1e-06))
    ):
        raise ValueError("Invalid predicted probabilities")
    if set(labels) != set(range(count)) or not set(predicted).issubset(range(count)):
        raise ValueError("All classes must occur in held-out labels")
    p, r, f, s = precision_recall_fscore_support(
        labels, predicted, labels=list(range(count)), zero_division=0
    )
    binary = np.eye(count)[labels]
    auc = [
        float(roc_auc_score(binary[:, j], probabilities[:, j])) for j in range(count)
    ]
    ap = [
        float(average_precision_score(binary[:, j], probabilities[:, j]))
        for j in range(count)
    ]
    return dict(
        accuracy=float(np.mean(labels == predicted)),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        macro_f1=float(f.mean()),
        weighted_f1=float(np.dot(f, s) / s.sum()),
        macro_roc_auc=float(np.mean(auc)),
        macro_average_precision=float(np.mean(ap)),
        loss=loss,
        confusion_matrix=confusion_matrix(
            labels, predicted, labels=list(range(count))
        ).tolist(),
        per_class=[
            dict(
                label=j,
                name=classes[j],
                precision=float(p[j]),
                recall=float(r[j]),
                f1=float(f[j]),
                support=int(s[j]),
                roc_auc=auc[j],
                average_precision=ap[j],
            )
            for j in range(count)
        ],
    )
