import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ("ER", "BA", "WS", "SBM", "HK")
CLASS_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
MODELS = (
    "GCN",
    "GAT",
    "GATV2",
    "GIN",
    "SAGE",
    "GTN",
    "SVM",
    "Graph2Vec_SVM",
    "Descriptor_SVM_tuned",
)
NAMES = {
    **dict(zip(MODELS, MODELS)),
    "GATV2": "GATv2",
    "SAGE": "GraphSAGE",
    "SVM": "SVM (fixed)",
    "Graph2Vec_SVM": "Graph2Vec-SVM",
    "Descriptor_SVM_tuned": "SVM (tuned)",
}
TABLE_NAMES = {
    **NAMES,
    "SVM": "SVM-F",
    "Graph2Vec_SVM": "G2V-SVM",
    "Descriptor_SVM_tuned": "SVM-T",
}
TABLE_KEY = (
    " SVM-F: fixed descriptor SVM; SVM-T: tuned descriptor SVM; G2V-SVM: Graph2Vec-SVM."
)
COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#333333",
    "#6A3D9A",
    "#8C510A",
)
STYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    "-",
    (0, (4, 1, 1, 1)),
)


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "savefig.facecolor": "white",
        }
    )


def save(fig, output, name):
    fig.savefig(
        output / name,
        metadata={"Creator": None, "Producer": None, "CreationDate": None},
    )
    plt.close(fig)


def confusion(frames, output):
    data = frames["confusion_summary"]
    fig, axes = plt.subplots(3, 3, figsize=(5.5, 6.15))
    fig.subplots_adjust(
        left=0.068, right=0.993, top=0.95, bottom=0.11, wspace=0.3, hspace=0.36
    )
    for index, (model, ax) in enumerate(zip(MODELS, axes.flat)):
        subset = data.loc[data.architecture == model]
        if len(subset) != 25 or not (subset.row_normalized_n == 5).all():
            raise ValueError(f"Incomplete five-seed confusion matrix for {model}")

        def matrix(column):
            return (
                subset.pivot(
                    index="true_label", columns="predicted_label", values=column
                )
                .loc[range(5), range(5)]
                .to_numpy(float)
                * 100
            )

        mean, sd = (matrix("row_normalized_mean"), matrix("row_normalized_sd"))
        if not np.allclose(mean.sum(axis=1), 100):
            raise ValueError("Confusion matrix rows must each sum to 100%")
        im = ax.imshow(mean, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for row in range(5):
            for col in range(5):
                ax.text(
                    col,
                    row,
                    f"{mean[row, col]:.1f}\n±{sd[row, col]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.9,
                    linespacing=1.05,
                    color="white" if mean[row, col] >= 60 else "#202020",
                )
        ax.set_title(NAMES[model], pad=4, fontsize=7.7)
        ax.set_xticks(range(5), CLASSES, fontsize=6.4)
        ax.set_yticks(range(5), CLASSES, fontsize=6.4)
        ax.tick_params(length=0, pad=2)
        if index >= 6:
            ax.set_xlabel("Predicted class", fontsize=7, labelpad=4)
        if index % 3 == 0:
            ax.set_ylabel("True class", fontsize=7, labelpad=3)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
    fig.text(
        0.5,
        0.989,
        "Five paired repetitions; cells show mean % ± sample SD",
        ha="center",
        va="top",
        fontsize=7.3,
    )
    cax = fig.add_axes([0.3, 0.033, 0.43, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    save(fig, output, "confusion_all.pdf")


def curve_plot(frames, output, model=None):
    curves = frames["display_curves_summary"]
    metrics = frames["metrics_summary"].set_index("architecture")
    class_metrics = frames["class_metrics_summary"].set_index(
        ["architecture", "class_label"]
    )
    classwise = model is not None
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 4.25 if not classwise else 3.7))
    fig.subplots_adjust(
        left=0.095,
        right=0.99,
        bottom=0.4 if not classwise else 0.33,
        top=0.92 if not classwise else 0.86,
        wspace=0.32,
    )
    categories = list(range(5)) if classwise else list(MODELS)
    colors = CLASS_COLORS if classwise else COLORS
    styles = STYLES[:5] if classwise else STYLES
    titles = (
        ("(a) ROC: AUC mean ± SD", "(b) Precision-recall: AP mean ± SD")
        if classwise
        else ("(a) Macro ROC: AUC mean ± SD", "(b) Macro PR: AP mean ± SD")
    )
    for ax, kind, metric, title in zip(
        axes,
        ("roc", "pr"),
        ("roc_auc_macro_ovr", "average_precision_macro_ovr"),
        titles,
    ):
        for category, color, style in zip(categories, colors, styles):
            architecture = model if classwise else category
            class_label = str(category) if classwise else "macro"
            block = curves.loc[
                (curves.architecture == architecture)
                & (curves.class_label.astype(str) == class_label)
                & (curves.curve == kind)
            ].sort_values("x")
            if len(block) != 201 or not (block.y_n == 5).all():
                raise ValueError(
                    f"Incomplete five-seed {architecture} {class_label} {kind} curve"
                )
            x = block.x.to_numpy(float)
            y, sd = (block.y_mean.to_numpy(float), block.y_sd.to_numpy(float))
            if not np.allclose(x, np.linspace(0, 1, 201)):
                raise ValueError("Unexpected probability-ranking display grid")
            if classwise:
                row = class_metrics.loc[architecture, category]
                key = "roc_auc_ovr" if kind == "roc" else "average_precision_ovr"
                name = CLASSES[category]
            else:
                row, key, name = (
                    metrics.loc[architecture],
                    metric,
                    NAMES[architecture],
                )
            label = f"{name}: {row[key + '_mean']:.4f} ± {row[key + '_sd']:.4f}"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=style,
                lw=1.0,
                label=label,
                drawstyle="steps-pre" if kind == "pr" else "default",
            )
            ax.fill_between(
                x,
                np.clip(y - sd, 0, 1),
                np.clip(y + sd, 0, 1),
                color=color,
                alpha=0.085,
                linewidth=0,
                step="pre" if kind == "pr" else None,
            )
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="0.65", lw=0.6, linestyle="--", zorder=0)
        else:
            ax.axhline(0.2, color="0.65", lw=0.6, linestyle="--", zorder=0)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.015),
            xlabel="False-positive rate" if kind == "roc" else "Recall",
            ylabel="True-positive rate" if kind == "roc" else "Precision",
        )
        ax.set_title(title, fontsize=7.3, pad=8)
        ticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ["0", "0.25", "0.5", "0.75", "1"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.tick_params(labelsize=7, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.06, -0.235),
            fontsize=6.55,
            frameon=False,
            handlelength=2.3,
            labelspacing=0.36,
            borderpad=0,
            handletextpad=0.5,
        )
    if classwise:
        fig.suptitle(NAMES[model] + ": class-wise held-out ranking", fontsize=9, y=0.98)
    name = f"roc_pr_{model}.pdf" if classwise else "roc_pr_macro_compact.pdf"
    save(fig, output, name)


def cell(row, key, scale=1, digits=2):
    return f"${row[key + '_mean'] * scale:.{digits}f}\\pm{row[key + '_sd'] * scale:.{digits}f}$"


def latex_table(path, caption, label, headings, rows, size="small", alignment=None):
    alignment = alignment or "l" + "r" * (len(headings) - 1)
    text = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\" + size,
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{" + caption + "}\\label{" + label + "}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + alignment + "@{}}",
        "\\toprule",
        " & ".join(headings) + " \\\\",
        "\\midrule",
    ]
    text += [" & ".join(row) + " \\\\" for row in rows]
    text += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    path.write_text("\n".join(text))


def tables(frames, runs, output):
    output.mkdir(exist_ok=True)
    aggregate = frames["metrics_summary"].set_index("architecture")
    rows = [
        [TABLE_NAMES[m]]
        + [
            cell(aggregate.loc[m], key, 100)
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        ]
        for m in MODELS
    ]
    latex_table(
        output / "test_metrics.tex",
        "Held-out performance (\\%): mean $\\pm$ sample standard deviation over five paired repetitions. Precision, recall and F1 are unweighted macro averages across the five equally represented classes. Fixed and tuned SVMs use the same selected descriptors; Graph2Vec-SVM uses learned structural embeddings."
        + TABLE_KEY,
        "tab:accuracy",
        ["Model", "Accuracy", "Precision", "Recall", "F1"],
        rows,
    )
    classes = frames["class_metrics_summary"].set_index(["architecture", "class_label"])
    for metric, label in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1score"),
    ):
        rows = [
            [TABLE_NAMES[m]] + [cell(classes.loc[m, k], metric, 100) for k in range(5)]
            for m in MODELS
        ]
        latex_table(
            output / f"{metric}_by_class.tex",
            "Class-wise "
            + ("F1-score" if metric == "f1" else metric)
            + " (\\%): mean $\\pm$ sample standard deviation across five repetitions. Each test partition contains 60 graphs per class."
            + TABLE_KEY,
            "tab:" + label,
            ["Model", *CLASSES],
            rows,
            "footnotesize",
        )
    rows = [
        [TABLE_NAMES[m]]
        + [
            cell(aggregate.loc[m], key, digits=4)
            for key in (
                "roc_auc_macro_ovr",
                "average_precision_macro_ovr",
                "probability_log_loss",
            )
        ]
        for m in MODELS
    ]
    latex_table(
        output / "ranking_loss.tex",
        "Probability-based test metrics: mean $\\pm$ sample standard deviation over five repetitions. ROC-AUC and average precision (AP) use one-vs-rest scores and an unweighted class mean. Probability log loss is reported on its natural scale; smaller values indicate lower loss."
        + TABLE_KEY,
        "tab:ranking_loss",
        ["Model", "Macro ROC-AUC", "Macro AP", "Probability log loss"],
        rows,
    )
    selected = {
        (r.architecture, r.seed): r.parameters
        for r in runs
        if r.architecture in MODELS[-2:]
    }
    rows = []
    for model in MODELS[-2:]:
        p = [selected[model, seed] for seed in range(42, 47)]
        rows.append(
            [
                TABLE_NAMES[model],
                ", ".join((f"{x['C']:g}" for x in p)),
                ", ".join((f"{x['gamma_multiplier']:g}" for x in p)),
            ]
        )
    latex_table(
        output / "selected_svm_parameters.tex",
        "Validation-selected SVM settings, in seed order 42--46. Both representations use the same 25-candidate grid. The multiplier scales $\\gamma_0=1/(d\\,\\operatorname{Var}(X_{\\mathrm{train}}))$ after training-only standardisation. Exact $\\gamma$ values and validation scores are supplied with the results."
        + TABLE_KEY,
        "tab:svm_parameters",
        ["Representation", "$C$ (42--46)", "$\\gamma/\\gamma_0$ (42--46)"],
        rows,
        "footnotesize",
    )
    intervals = frames["paired_accuracy_intervals"]
    intervals = intervals.loc[intervals.n_denominator == 1400]
    if len(intervals) != 36 or set(intervals.family_size) != {36}:
        raise ValueError("The expanded comparison family must contain all 36 pairs")
    rows = []

    def interval(low, high):
        return (
            "not estimable"
            if pd.isna(low)
            else f"$[{100 * low:.2f},\\,{100 * high:.2f}]$"
        )

    for row in intervals.itertuples():
        rows.append(
            [
                TABLE_NAMES[row.architecture_a],
                TABLE_NAMES[row.architecture_b],
                f"${100 * row.accuracy_difference_mean:.2f}$",
                interval(row.pointwise95_low, row.pointwise95_high),
                interval(row.bonferroni95_low, row.bonferroni95_high),
            ]
        )
    latex_table(
        output / "paired_intervals.tex",
        "Paired accuracy differences (A minus B), in percentage points. Intervals use the corrected-resampling approximation with four degrees of freedom and ratio $300/1400$. The adjusted column applies Bonferroni correction to all 36 comparisons. Coverage is nominal and approximate."
        + TABLE_KEY,
        "tab:paired_accuracy_intervals",
        [
            "A",
            "B",
            "Mean difference",
            "Pointwise 95\\% interval",
            "Adjusted 95\\% interval",
        ],
        rows,
        "footnotesize",
        "llrrr",
    )
