import argparse
from pathlib import Path
import subprocess
import sys
from support import load_protocol

ROOT = Path(__file__).resolve().parent


def commands(args):
    suite = load_protocol(args.software_test)
    seeds = args.seeds or suite["seeds"]
    architectures = args.architectures or suite["architectures"]
    conditions = args.conditions or suite["conditions"]
    graph_modes = args.graph_modes or suite["graph_modes"]
    for requested, permitted, name in (
        (seeds, suite["seeds"], "seeds"),
        (architectures, suite["architectures"], "architectures"),
        (conditions, suite["conditions"], "conditions"),
        (graph_modes, suite["graph_modes"], "graph modes"),
    ):
        if len(set(requested)) != len(requested) or not set(requested).issubset(
            permitted
        ):
            raise ValueError(f"Invalid or duplicate {name}")
    common = [
        "--dataset",
        str(args.dataset),
        "--output",
        str(args.output),
        "--prepared",
        str(args.prepared or args.output / "prepared"),
    ]
    if args.software_test:
        common.append("--software-test")
    neural = ["--device", args.device, "--threads", str(args.threads)]
    for seed in seeds:
        if args.stage in {"main", "all"}:
            for architecture in architectures:
                yield [
                    sys.executable,
                    str(ROOT / "run_main.py"),
                    *common,
                    *neural,
                    "--seed",
                    str(seed),
                    "--architecture",
                    architecture,
                ]
            yield [
                sys.executable,
                str(ROOT / "run_svm.py"),
                *common,
                "--seed",
                str(seed),
            ]
        if args.stage in {"ablation", "all"}:
            for architecture in architectures:
                for condition in conditions:
                    for mode in graph_modes:
                        yield [
                            sys.executable,
                            str(ROOT / "run_ablation.py"),
                            *common,
                            *neural,
                            "--seed",
                            str(seed),
                            "--architecture",
                            architecture,
                            "--condition",
                            condition,
                            "--graph-mode",
                            mode,
                        ]


def main():
    parser = argparse.ArgumentParser(
        description="Run the main graph-classification comparison and node-attribute ablations."
    )
    parser.add_argument("--stage", choices=["main", "ablation", "all"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--graph-modes", nargs="+")
    parser.add_argument("--software-test", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="List selected tasks without executing"
    )
    args = parser.parse_args()
    work = list(commands(args))
    print(f"{len(work)} selected workflows", flush=True)
    for index, command in enumerate(work, 1):
        print(f"[{index}/{len(work)}] " + " ".join(command), flush=True)
        if not args.list:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
