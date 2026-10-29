import argparse
import os
import shutil
import signal
from pathlib import Path
from core import read, verify_run, write, commit, done_valid, digest


def execute(run, task_id, device="cpu", smoke=False):
    run = Path(run).resolve()
    manifest = verify_run(run)
    if bool(manifest.get("smoke", False)) != smoke:
        raise ValueError("Smoke mode must agree with the prepared run")
    task = next((task for task in read(run / "tasks.json") if task["id"] == task_id))
    if done_valid(run / "tasks" / task_id, True):
        return
    if not all((done_valid(run / "tasks" / dep, True) for dep in task["deps"])):
        raise RuntimeError("Task dependencies are incomplete")
    if task["kind"] == "svm":
        from baseline import run_svm

        run_svm(task, run, smoke)
        return
    from learning import fit, choose

    if task["kind"] == "final" and task["mode"] == "paired":
        source = (
            run
            / "tasks"
            / task_id.rsplit("/", 1)[0]
            / f"independent_{task['initialization']}"
        )
        if done_valid(source, True):
            metrics = read(source / "metrics.json")
            if choose(task, run) == metrics["config_index"]:
                dest = run / "tasks" / task_id
                dest.mkdir(parents=True, exist_ok=True)
                files = read(source / "complete.json")["files"]
                for name in files:
                    if name not in ("metrics.json", "selection_frozen.json"):
                        shutil.copy2(source / name, dest / name)
                metrics["reused_from"] = str(source.relative_to(run / "tasks"))
                metrics["reused_source_identity"] = metrics["identity"]
                metrics["task"] = task
                metrics["seconds"] = 0.0
                cfg = read(run / "data" / task["dataset"] / f"fold_{task['fold']}.json")
                metrics["identity"] = digest(
                    dict(
                        task=task,
                        params=metrics["params"],
                        data=cfg,
                        source=manifest["source_hashes"],
                    )
                )
                write(dest / "metrics.json", metrics)
                selection = read(source / "selection_frozen.json")
                selection["dependencies"] = task["deps"]
                selection["reused_from"] = metrics["reused_from"]
                write(dest / "selection_frozen.json", selection)
                commit(dest, list(files))
                print("Reused identical fit:", metrics["reused_from"], flush=True)
                return
    fit(task, run, device, smoke)


def main():
    parser = argparse.ArgumentParser(
        description="Execute one prepared matched-control experiment task."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    def stop(*_):
        os.environ["CONTROLS_STOP"] = "1"

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    execute(args.run, args.id, args.device, args.smoke)


if __name__ == "__main__":
    main()
