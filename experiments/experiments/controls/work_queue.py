import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from core import ROOT, read, write, lock, verify_run, done_valid


def preflight(device):
    from learning import setup

    setup(device)


def snapshot(run):
    run = Path(run)
    tasks = read(run / "tasks.json")
    done = {task["id"] for task in tasks if done_valid(run / "tasks" / task["id"])}
    counts = collections.Counter()
    details = []
    for task in tasks:
        folder = run / "tasks" / task["id"]
        if task["id"] in done:
            state = "complete"
        else:
            with lock(folder / "task.lock") as held:
                if held is None:
                    state = "running"
                elif (folder / "failure.json").exists():
                    state = "failed"
                else:
                    state = (
                        "ready"
                        if all((dep in done for dep in task["deps"]))
                        else "waiting_dependencies"
                    )
        counts[state] += 1
        details.append(dict(id=task["id"], state=state))
    return dict(
        total=len(tasks),
        counts=dict(counts),
        all_complete=len(done) == len(tasks),
        tasks=details,
    )


def work(run, device="cpu", smoke=False, poll_seconds=2):
    run = Path(run).resolve()
    manifest = verify_run(run)
    if bool(manifest.get("smoke", False)) != smoke:
        raise ValueError("Smoke mode must agree with the prepared run")
    tasks = read(run / "tasks.json")
    if all((done_valid(run / "tasks" / task["id"], True) for task in tasks)):
        print("ALL WORK COMPLETE.", flush=True)
        return 0
    preflight(device)
    stop = False
    child = None

    def stopping(*_):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)

    old_handlers = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)
    }
    for sig in old_handlers:
        signal.signal(sig, stopping)
    try:
        with lock(run / "workers.lock", shared=True) as gate:
            if gate is None:
                raise RuntimeError(
                    "Queue maintenance is running; retry the worker shortly"
                )
            ordered = sorted(
                tasks,
                key=lambda task: (
                    task["kind"] == "trial",
                    task.get("dataset") != "synthetic",
                    task["id"],
                ),
            )
            while not stop:
                completed = {
                    task["id"]
                    for task in tasks
                    if done_valid(run / "tasks" / task["id"])
                }
                if len(completed) == len(tasks):
                    for task in tasks:
                        done_valid(run / "tasks" / task["id"], True)
                    print("ALL WORK COMPLETE.", flush=True)
                    return 0
                claimed = False
                for task in ordered:
                    if stop:
                        break
                    folder = run / "tasks" / task["id"]
                    if task["id"] in completed or not all(
                        (dep in completed for dep in task["deps"])
                    ):
                        continue
                    with lock(folder / "task.lock") as held:
                        if (
                            held is None
                            or done_valid(folder)
                            or (folder / "failure.json").exists()
                        ):
                            continue
                        if stop:
                            break
                        claimed = True
                        write(
                            folder / "owner.json",
                            dict(pid=os.getpid(), started=time.time(), device=device),
                        )
                        print("CLAIM", task["id"], flush=True)
                        command = [
                            sys.executable,
                            "-u",
                            str(ROOT / "run_task.py"),
                            "--run",
                            str(run),
                            "--id",
                            task["id"],
                            "--device",
                            device,
                        ]
                        if smoke:
                            command.append("--smoke")
                        try:
                            with open(
                                folder / "execution.out", "ab", buffering=0
                            ) as stdout, open(
                                folder / "execution.err", "ab", buffering=0
                            ) as stderr:
                                child = subprocess.Popen(
                                    command,
                                    stdout=stdout,
                                    stderr=stderr,
                                    pass_fds=(held.fileno(), gate.fileno()),
                                )
                                if stop and child.poll() is None:
                                    child.send_signal(signal.SIGUSR1)
                                code = child.wait()
                                child = None
                            if done_valid(folder):
                                print("DONE", task["id"], flush=True)
                            elif code == 75 or stop:
                                print(
                                    "INTERRUPTED; checkpoint retained:",
                                    task["id"],
                                    flush=True,
                                )
                                return 75
                            else:
                                write(
                                    folder / "failure.json",
                                    dict(
                                        returncode=code,
                                        time=time.time(),
                                        task=task["id"],
                                    ),
                                )
                                print(
                                    "FAILED",
                                    task["id"],
                                    "see execution.err",
                                    flush=True,
                                )
                        finally:
                            (folder / "owner.json").unlink(missing_ok=True)
                    break
                if not claimed and (not stop):
                    state = snapshot(run)
                    if state["all_complete"]:
                        for task in tasks:
                            done_valid(run / "tasks" / task["id"], True)
                        return 0
                    if state["counts"].get("ready", 0):
                        continue
                    if state["counts"].get("running", 0):
                        time.sleep(poll_seconds)
                        continue
                    print("No executable tasks remain:", state["counts"], flush=True)
                    return 2
            return 75
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def retry_failures(run):
    run = Path(run).resolve()
    verify_run(run)
    with lock(run / "workers.lock") as held:
        if held is None:
            raise RuntimeError("Stop all workers before resetting failed tasks")
        failures = list((run / "tasks").rglob("failure.json"))
        for failure in failures:
            with lock(failure.parent / "task.lock") as task_lock:
                if task_lock is None:
                    raise RuntimeError(
                        "A task is still running: " + str(failure.parent)
                    )
                failure.rename(
                    failure.with_name(f"failure.resolved.{time.time_ns()}.json")
                )
    print(f"Reset {len(failures)} failures; checkpoints and completed tasks retained.")


def main():
    parser = argparse.ArgumentParser(
        description="Run, inspect, and resume the matched-control experiment queue."
    )
    parser.add_argument(
        "command", choices=["worker", "status", "is-complete", "retry-failures"]
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        return work(args.run, args.device, args.smoke)
    if args.command == "retry-failures":
        retry_failures(args.run)
        return 0
    verify_run(args.run)
    state = snapshot(args.run)
    if args.command == "is-complete":
        return 0 if state["all_complete"] else 1
    print(
        json.dumps(
            {key: value for key, value in state.items() if key != "tasks"}, indent=2
        )
    )
    for task in state["tasks"]:
        if task["state"] in ("failed", "running"):
            print(task["state"].upper(), task["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
