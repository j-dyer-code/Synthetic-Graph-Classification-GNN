import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, read, sha, source_hashes, write
from protocol import tasks
from test_extension import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError("Choose an empty smoke-run directory")
    fixture(run)
    selected = [
        task
        for task in tasks(["synthetic"], smoke=True)
        if task["kind"] == "svm"
        or task["model"]["arch"] in ("SAGE", "DeepSets", "GraphMLP")
    ]
    write(run / "tasks.json", selected)
    write(
        run / "manifest.json",
        dict(
            smoke=True,
            purpose="software_test",
            source_hashes=source_hashes(),
            tasks_sha256=sha(run / "tasks.json"),
            synthetic_scope="software_test",
            data_hashes={
                str(path.relative_to(run)): sha(path)
                for path in (run / "data").rglob("*.json")
            },
        ),
    )
    command = [
        sys.executable,
        str(ROOT / "work_queue.py"),
        "worker",
        "--run",
        str(run),
        "--device",
        "cpu",
        "--smoke",
    ]
    with open(run / "worker_1.log", "w") as first, open(
        run / "worker_2.log", "w"
    ) as second:
        workers = [
            subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            for stream in (first, second)
        ]
        codes = [worker.wait() for worker in workers]
    if codes != [0, 0]:
        raise RuntimeError(f"Worker failure {codes}; inspect worker/task logs in {run}")
    subprocess.run(
        [sys.executable, str(ROOT / "analyse.py"), "--run", str(run), "--smoke"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "work_queue.py"),
            "worker",
            "--run",
            str(run),
            "--device",
            "cuda",
            "--smoke",
        ],
        check=True,
    )
    reused = sum(
        (
            bool(read(run / "tasks" / task["id"] / "metrics.json").get("reused_from"))
            for task in selected
        )
    )
    if len(selected) != 11 or reused != 2:
        raise AssertionError((len(selected), reused))
    print(
        json.dumps(
            dict(
                status="passed",
                logical_tasks=len(selected),
                reused_fits=reused,
                concurrent_workers=2,
            )
        )
    )


if __name__ == "__main__":
    main()
