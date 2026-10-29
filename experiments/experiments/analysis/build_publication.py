import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import metrics
import publication_style as style
from result_io import (
    verify_prepared,
    read,
    prediction_frame,
    main_run_path,
    validate_test_identity,
    validate_saved_test,
    verified_sources,
)


def load_runs(root, baselines, prepared):
    runs, files = ([], [])
    for model in style.MODELS:
        for seed in range(42, 47):
            path = main_run_path(root, baselines, model, seed)
            pred_path = path / "test_predictions.csv"
            prediction = prediction_frame(pred_path)
            validate_test_identity(prediction, prepared, seed)
            if model in style.MODELS[:6] and (
                not np.array_equal(
                    prediction.predicted_label, prediction.probability_argmax
                )
            ):
                raise ValueError(
                    f"GNN hard predictions must equal probability argmax: {path}"
                )
            aggregate, per_class, confusion, curves = metrics.classification_metrics(
                prediction
            )
            saved = read(path / "metrics.json")
            validate_saved_test(saved["test"], aggregate, confusion, path)
            history_path = path / "history.json"
            history = read(history_path) if history_path.exists() else []
            parameter_path = path / (
                "best_parameters.json"
                if model in style.MODELS[:6]
                else "parameters.json"
            )
            if model == "SVM" and (not parameter_path.exists()):
                parameter_path = path / "best_parameters.json"
            parameters = read(parameter_path) if parameter_path.exists() else {}
            runs.append(
                metrics.Run(
                    seed=seed,
                    architecture=model,
                    path=path,
                    config={},
                    metrics=saved,
                    tuning={},
                    parameters=parameters,
                    history=history,
                    predictions=prediction,
                    aggregate=aggregate,
                    per_class=per_class,
                    confusion=confusion,
                    curves=curves,
                )
            )
            required = [pred_path, path / "metrics.json"]
            if history_path.exists():
                required.append(history_path)
            if parameter_path.exists():
                required.append(parameter_path)
            files.extend(verified_sources(path, required))
    return (runs, files)


def load_ablation(root, prepared):
    rows, histories, sources = ([], [], [])
    conditions = (
        "topology",
        "constant",
        "independent",
        "permuted",
        "mixture_025",
        "mixture_050",
        "mixture_075",
        "community",
    )
    for seed in range(42, 47):
        for model in style.MODELS[:6]:
            for condition in conditions:
                for mode in ("on", "off"):
                    path = (
                        root
                        / "runs/ablation"
                        / f"seed_{seed}"
                        / model
                        / f"{condition}__graph_{mode}"
                    )
                    prediction = prediction_frame(path / "test_predictions.csv")
                    validate_test_identity(prediction, prepared, seed)
                    if not np.array_equal(
                        prediction.predicted_label, prediction.probability_argmax
                    ):
                        raise ValueError(
                            f"Ablation hard predictions must equal probability argmax: {path}"
                        )
                    aggregate, _, confusion, _ = metrics.classification_metrics(
                        prediction
                    )
                    saved = read(path / "metrics.json")
                    validate_saved_test(saved["test"], aggregate, confusion, path)
                    base = dict(
                        seed=seed, model=model, condition=condition, graph_mode=mode
                    )
                    rows.append(dict(**base, **aggregate))
                    histories.extend(
                        (
                            dict(
                                seed=seed,
                                architecture=model,
                                condition=condition,
                                graph_mode=mode,
                                **{k: v for k, v in r.items() if k != "calibration"},
                            )
                            for r in read(path / "history.json")
                        )
                    )
                    sources.extend(
                        verified_sources(
                            path,
                            [
                                path / name
                                for name in (
                                    "test_predictions.csv",
                                    "metrics.json",
                                    "history.json",
                                )
                            ],
                        )
                    )
    return (pd.DataFrame(rows), pd.DataFrame(histories), sources)


def final_fit_runtime(runs, baselines, output):
    def cell(values, scale=1, digits=2):
        a = np.asarray(values, dtype=float) * scale
        if a.shape != (5,) or not np.isfinite(a).all():
            raise ValueError("Five finite per-repetition measurements are required")
        return f"${a.mean():.{digits}f}\\pm{a.std(ddof=1):.{digits}f}$"

    size_rows, time_rows, timing_records = ([], [], [])
    for model in style.MODELS:
        subset = sorted(
            [r for r in runs if r.architecture == model], key=lambda r: r.seed
        )
        if model in style.MODELS[:6]:
            size_rows.append(
                [
                    style.TABLE_NAMES[model],
                    cell([r.metrics["parameter_count"] for r in subset], 1 / 1000),
                    cell([r.metrics["best_epoch"] for r in subset], digits=1),
                    cell([r.metrics["epochs_executed"] for r in subset], digits=1),
                ]
            )
        elif model == "SVM":
            size_rows.append([style.TABLE_NAMES[model], "--", "--", "--"])
        search = (
            "--"
            if model == "SVM"
            else cell(
                [r.metrics["timings"]["tuning_seconds"] for r in subset],
                1 / 60,
                4 if model in style.MODELS[-2:] else 2,
            )
        )
        fit_key = (
            "final_training_seconds" if model in style.MODELS[:6] else "svm_fit_seconds"
        )
        time_rows.append(
            [
                style.TABLE_NAMES[model],
                search,
                cell(
                    [r.metrics["timings"][fit_key] for r in subset],
                    digits=1 if model in style.MODELS[:6] else 4,
                ),
                cell(
                    [r.metrics["timings"]["test_inference_seconds"] for r in subset],
                    digits=3 if model in style.MODELS[:6] else 4,
                ),
            ]
        )
        for r in subset:
            timing_records.append(
                dict(architecture=model, seed=r.seed, **r.metrics["timings"])
            )
    fitting, inference = ([], [])
    representation_files = []
    for seed in range(42, 47):
        path = baselines / "runs" / f"seed_{seed}/representation/model_info.json"
        info = read(path)
        representation_files.extend(verified_sources(path.parent, [path]))
        fitting.append(info["training_checkpoint"]["cumulative_fit_seconds"])
        inference.append(sum(info["inference_seconds_per_graph"]))
    caption = (
        "Model size and measured computation: mean $\\pm$ sample standard deviation over five repetitions. GNN search is active wall time; final fitting includes training, BatchNorm calibration and validation epochs, including calibration-record writing but excluding initialisation and training-checkpoint I/O. SVM final fitting includes probability calibration. Classifier test times exclude serialisation and representation construction. Graph2Vec fitting covers training-vocabulary construction and completed epochs, excluding initialisation and checkpoint I/O; embedding inference with vocabulary-coverage diagnostics covers all 2,000 graphs. Execution settings are recorded per run. Graph generation, descriptor extraction and feature selection are excluded."
        + style.TABLE_KEY
    )
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{" + caption + "}\\label{tab:training_time}",
    ]
    panels = [
        (
            "(a) Model size and GNN final-training epochs",
            ["Model", "Params. ($10^3$)", "Best epoch", "Stop epoch"],
            size_rows,
        ),
        (
            "(b) Classifier phases",
            ["Model", "Search (min)", "Final fit (s)", "Test (s)"],
            time_rows,
        ),
        (
            "(c) Graph2Vec representation construction",
            ["Phase", "Time (s)"],
            [
                ["Training vocabulary and embedding fitting", cell(fitting)],
                ["Inference with coverage diagnostics (all graphs)", cell(inference)],
            ],
        ),
    ]
    for title, headings, rows in panels:
        width = len(headings)
        lines += [
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l"
            + "r" * (width - 1)
            + "@{}}",
            "\\toprule",
            "\\multicolumn{" + str(width) + "}{l}{\\textit{" + title + "}} \\\\",
            "\\midrule",
            " & ".join(headings) + " \\\\",
            "\\midrule",
        ]
        lines += [" & ".join(row) + " \\\\" for row in rows]
        lines += ["\\bottomrule", "\\end{tabular*}", "\\par\\smallskip"]
    lines += ["\\end{table}", ""]
    (output / "tables/runtime.tex").write_text("\n".join(lines))
    pd.DataFrame(timing_records).to_csv(
        output / "data/per_run_timings.csv", index=False
    )
    return representation_files


def parameters(runs, output):
    rows, raw = ([], [])
    for model in style.MODELS[:6]:
        subset = sorted(
            [r for r in runs if r.architecture == model], key=lambda r: r.seed
        )
        row = [
            style.TABLE_NAMES[model],
            ", ".join((str(r.parameters["hidden_channels"]) for r in subset)),
        ]
        for key, factor, digits in [("learning_rate", 1000, 3), ("dropout", 1, 3)]:
            a = np.array([r.parameters[key] for r in subset]) * factor
            row.append(f"${a.mean():.{digits}f}\\pm{a.std(ddof=1):.{digits}f}$")
        rows.append(row)
        raw += [dict(architecture=model, seed=r.seed, **r.parameters) for r in subset]
    style.latex_table(
        output / "tables/selected_parameters.tex",
        "Selected GNN hyperparameters. Widths are listed in seed order 42--46; learning rate and dropout are summarised as mean $\\pm$ sample standard deviation across selected configurations. Exact per-seed values are supplied with the results.",
        "tab:candidate_models",
        ["Model", "Widths (42--46)", "Learning rate ($10^{-3}$)", "Dropout"],
        rows,
    )
    pd.DataFrame(raw).to_csv(output / "data/selected_parameters.csv", index=False)


def ablation_table(frame, output):
    if (
        len(frame) != 480
        or frame.duplicated(["model", "seed", "condition", "graph_mode"]).any()
    ):
        raise ValueError("Expected 480 distinct ablation evaluations")
    summary = frame.groupby(["condition", "graph_mode", "model"]).accuracy.agg(
        ["mean", "std", "count"]
    )
    if not (summary["count"] == 5).all():
        raise ValueError("Every ablation condition requires five paired evaluations")
    order = ["GCN", "GAT", "GATV2", "SAGE", "GIN", "GTN"]
    conditions = [
        ("topology", "Topological"),
        ("constant", "Constant"),
        ("independent", "Independent"),
        ("permuted", "Permuted"),
        ("mixture_025", "Mix. $\\alpha=0.25$"),
        ("mixture_050", "Mix. $\\alpha=0.50$"),
        ("mixture_075", "Mix. $\\alpha=0.75$"),
        ("community", "Community"),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Node-attribute ablation: test accuracy (\\%) as mean $\\pm$ sample standard deviation across five paired seeds. Graph descriptors are zeroed in panel (a) and retained in panel (b). All conditions use the same fixed training configuration.}\\label{tab:attribute_ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "Condition & GCN & GAT & GATv2 & GraphSAGE & GIN & GTN \\\\",
        "\\midrule",
    ]
    for mode, title in [
        ("off", "(a) Graph descriptors absent"),
        ("on", "(b) Graph descriptors present"),
    ]:
        if mode == "on":
            lines.append("\\midrule")
        lines.append("\\multicolumn{7}{l}{\\textit{" + title + "}} \\\\")
        for condition, label in conditions:
            cells = [label]
            for model in order:
                row = summary.loc[condition, mode, model]
                cells.append(
                    f"${100 * row['mean']:.1f}\\!\\pm\\!{100 * row['std']:.1f}$"
                )
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    (output / "tables/ablation_table.tex").write_text("\n".join(lines))
    summary.reset_index().to_csv(
        output / "data/ablation_accuracy_summary.csv", index=False
    )


def build(args):
    root, baselines, output = (
        args.results.resolve(),
        args.baselines.resolve(),
        args.output.resolve(),
    )
    prepared = (args.prepared or root / "prepared").resolve()
    verify_prepared(prepared)
    runs, sources = load_runs(root, baselines, prepared)
    ablation, histories, ablation_sources = load_ablation(root, prepared)
    sources.extend(ablation_sources)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    classes = [
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
        "stochastic_block_model",
        "holme_kim",
    ]
    frames = metrics.collect_frames(runs, classes)
    if (
        len(frames["metrics_summary"]) != 9
        or not (frames["metrics_summary"].accuracy_n == 5).all()
    ):
        raise ValueError("All nine classifiers require five paired evaluations")
    style.configure()
    style.confusion(frames, output)
    style.curve_plot(frames, output)
    style.tables(frames, runs, output / "tables")
    parameters(runs, output)
    sources.extend(final_fit_runtime(runs, baselines, output))
    ablation_table(ablation, output)
    ablation.to_csv(output / "data/ablation_per_run_metrics.csv", index=False)
    for name, frame in frames.items():
        frame.to_csv(output / "data" / f"{name}.csv", index=False)
    histories.to_csv(output / "data/ablation_histories.csv", index=False)
    summaries = []
    for key, group in histories.groupby(
        ["architecture", "condition", "graph_mode", "epoch"]
    ):
        row = dict(zip(["architecture", "condition", "graph_mode", "epoch"], key))
        row["n_support"] = len(group)
        for column in (
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
        ):
            row[column + "_mean"] = group[column].mean()
            row[column + "_sd"] = group[column].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(
        output / "data/ablation_history_summary.csv", index=False
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "main_runs": 45,
                "ablation_runs": 480,
                "figures": 2,
                "tables": len(list((output / "tables").glob("*.tex"))),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build publication tables and figures from verified benchmark and baseline results."
    )
    parser.add_argument(
        "--results", required=True, type=Path, help="Benchmark output directory"
    )
    parser.add_argument(
        "--baselines",
        required=True,
        type=Path,
        help="Tuned-SVM and Graph2Vec output directory",
    )
    parser.add_argument(
        "--prepared", type=Path, help="Prepared inputs (default: RESULTS/prepared)"
    )
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args())
