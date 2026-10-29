from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import socket
import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "joblib",
    "gensim",
    "smart-open",
    "wrapt",
    "threadpoolctl",
    "pandas",
    "matplotlib",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path, value):
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def atomic_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(value):
            raise ValueError(
                f"Changed experiment identity: {path}; use a new output directory"
            )
    else:
        atomic_json(path, value)


def stable_seed(master, *parts):
    return int.from_bytes(
        hashlib.sha256(canonical([int(master), *parts]).encode()).digest()[:4], "big"
    )


def source_identity():
    result = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".json", ".txt", ".sh", ".sbatch")
        and ("__pycache__" not in p.parts)
    }
    return result


def environment():
    return dict(
        python=platform.python_version(),
        software={name: version(name) for name in PACKAGES},
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        threads={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )


@contextmanager
def exclusive_lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Active process holds {path}") from error
        handle.seek(0)
        handle.truncate()
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def complete(folder, identity, purpose, files):
    folder = Path(folder)
    fingerprint = identity if isinstance(identity, str) else digest(identity)
    atomic_json(
        folder / "complete.json",
        dict(
            purpose=purpose,
            fingerprint=fingerprint,
            files={str(name): file_hash(folder / name) for name in sorted(files)},
        ),
    )


def verify_complete(folder, identity=None):
    folder = Path(folder)
    marker = json.loads((folder / "complete.json").read_text())
    if marker.get("purpose") not in ("research", "software_test") or not marker.get(
        "files"
    ):
        raise ValueError("Invalid completion marker")
    fingerprint = (
        identity
        if isinstance(identity, str)
        else digest(identity)
        if identity is not None
        else None
    )
    if fingerprint is not None and marker["fingerprint"] != fingerprint:
        raise ValueError("Completion identity mismatch")
    for name, expected in marker["files"].items():
        path = folder / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or (not path.resolve().is_relative_to(folder.resolve()))
        ):
            raise ValueError("Unsafe completion path")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing or changed completed file: {path}")
    return marker


def load_protocol(smoke=False):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if smoke:
        protocol["purpose"] = "software_test"
        protocol["embedding"].update(
            dimension=8, train_epochs=2, inference_epochs=2, min_count=1, sample=0.0
        )
        protocol["data"].update(
            graphs_per_class=10, split_sizes=dict(train=35, validation=5, test=10)
        )
    return protocol
