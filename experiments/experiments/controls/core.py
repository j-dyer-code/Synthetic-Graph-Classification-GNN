import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLISHED_IDENTITY = ROOT.parent / "data" / "published_identity.json"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write(path, value):
    atomic(path, canonical(value) + b"\n")


@contextlib.contextmanager
def lock(path, blocking=False, shared=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(stream, flags | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
            return
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_hashes():
    files = list(ROOT.glob("*.py")) + [ROOT / "public_data_checksums.json"]
    files += list((ROOT / "reference_models").glob("*.py"))
    files += list((ROOT / "reference_splits").rglob("*.json"))
    files += [PUBLISHED_IDENTITY]
    return {os.path.relpath(path, ROOT): sha(path) for path in sorted(files)}


def data_path(run, record):
    path = Path(record["path"])
    if path.is_absolute():
        raise ValueError("Prepared graph paths must be relative to the run directory")
    return (Path(run) / path).resolve()


def verify_run(run):
    from protocol import SETTINGS, tasks

    run = Path(run)
    manifest = read(run / "manifest.json")
    if manifest["source_hashes"] != source_hashes():
        raise RuntimeError(
            "Source changed after preparation; use the original code or prepare a new run"
        )
    if sha(run / "tasks.json") != manifest["tasks_sha256"]:
        raise RuntimeError("Task manifest changed")
    task_list = read(run / "tasks.json")
    if not manifest.get("smoke", False):
        if manifest["protocol"] != SETTINGS or task_list != tasks():
            raise RuntimeError(
                "The prepared run differs from the fixed 4000-task protocol"
            )
    ids = {task["id"] for task in task_list}
    if len(ids) != len(task_list):
        raise ValueError("Duplicate task IDs")
    for task in task_list:
        path = Path(task["id"])
        if path.is_absolute() or ".." in path.parts or (not set(task["deps"]) <= ids):
            raise ValueError("Invalid task ID or dependency")
    for name, expected in manifest.get("data_hashes", {}).items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or sha(path) != expected:
            raise RuntimeError("Prepared split or metadata changed: " + name)
    return manifest


def done_valid(folder, full=False):
    folder = Path(folder)
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    files = read(marker).get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(
            "Completion record must list at least one artifact: " + str(marker)
        )
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Completed artifact paths must be relative: " + name)
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise RuntimeError("Invalid completed artifact: " + str(path))
        if full and sha(path) != expected:
            raise RuntimeError("Completed artifact changed: " + str(path))
    return True


def commit(folder, names):
    folder = Path(folder)
    write(
        folder / "complete.json",
        {"files": {name: sha(folder / name) for name in names}},
    )
