import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from result_io import verify_prepared
from publication_style import CLASSES, CLASS_COLORS, configure, save, latex_table

FEATURES = {
    "graph": (
        "Degree Variance",
        "Assortativity",
        "Density",
        "Clustering",
        "Average Path Length",
    ),
    "node": ("Degree", "Pagerank", "K-Core", "Eigenvector", "Closeness"),
}
DISPLAY_FEATURES = {
    "graph": (
        "Degree variance",
        "Assortativity",
        "Density",
        "Transitivity",
        "Average path length",
    ),
    "node": (
        "Mean degree",
        "Mean PageRank",
        "Mean coreness",
        "Mean eigenvector centrality",
        "Mean closeness",
    ),
}
SHORT_FEATURES = {
    "graph": ("Deg. var.", "Assort.", "Density", "Trans.", "Path len."),
    "node": ("Degree", "PageRank", "K-core", "Eigen.", "Closen."),
}


def numeric_format(ax, axis="x"):
    target = ax.xaxis if axis == "x" else ax.yaxis
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    fmt.set_useOffset(False)
    target.set_major_formatter(fmt)
    target.set_major_locator(MaxNLocator(nbins=4))
    target.get_offset_text().set_fontsize(6.5)


def candidates(descriptors, level, output):
    fig, axes = plt.subplots(
        5, 2, figsize=(5.5, 7.3), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    fig.subplots_adjust(
        left=0.12, right=0.975, top=0.91, bottom=0.065, hspace=0.65, wspace=0.34
    )
    handles = [
        Line2D([0], [0], color=color, lw=1.5, label=name)
        for name, color in zip(CLASSES, CLASS_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=1.7,
        columnspacing=1.5,
    )
    axes[0, 0].set_title("Gaussian KDE (Scott bandwidth)", pad=10)
    axes[0, 1].set_title("Box plots", pad=10)
    prefix = "graph_" if level == "graph" else "node_mean_"
    for j, label in enumerate(DISPLAY_FEATURES[level]):
        col = f"{prefix}{j}"
        values = [
            descriptors.loc[descriptors.label == c, col].to_numpy(float)
            for c in range(5)
        ]
        all_values = np.concatenate(values)
        low, high = (float(all_values.min()), float(all_values.max()))
        span = high - low
        pad = 0.05 * span if span > 0 else max(abs(low) * 0.05, 1e-08)
        xlim = (low - pad, high + pad)
        grid = np.linspace(*xlim, 2048)
        left, right = axes[j]
        for sample, color in zip(values, CLASS_COLORS):
            tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(sample))))
            if len(sample) < 2 or np.ptp(sample) <= tolerance:
                left.axvline(float(sample[0]), color=color, lw=1.1, linestyle=":")
            else:
                try:
                    density = gaussian_kde(sample, bw_method="scott")(grid)
                except np.linalg.LinAlgError:
                    left.axvline(
                        float(sample.mean()), color=color, lw=1.1, linestyle=":"
                    )
                else:
                    left.plot(grid, density, color=color, lw=1.05)
        left.set(xlim=xlim, ylim=(0, None), xlabel=label, ylabel="Density")
        numeric_format(left, "x")
        numeric_format(left, "y")
        box = right.boxplot(
            values,
            vert=False,
            patch_artist=True,
            widths=0.6,
            whis=1.5,
            manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 0.75},
            whiskerprops={"linewidth": 0.65},
            capprops={"linewidth": 0.65},
            flierprops={"marker": ".", "markersize": 1.5, "alpha": 0.45},
        )
        for c, color in enumerate(CLASS_COLORS):
            box["boxes"][c].set(
                facecolor=color, edgecolor=color, alpha=0.45, linewidth=0.7
            )
            box["fliers"][c].set(markeredgecolor=color, markerfacecolor=color)
            for item in (
                box["whiskers"][2 * c : 2 * c + 2] + box["caps"][2 * c : 2 * c + 2]
            ):
                item.set_color(color)
        right.set(xlim=xlim, ylim=(5.6, 0.4), xlabel=label)
        right.set_yticks(range(1, 6), CLASSES, fontsize=7)
        numeric_format(right, "x")
        for ax in (left, right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(pad=2)
            ax.grid(axis="x", color="0.88", lw=0.35)
            ax.set_axisbelow(True)
    save(fig, output, f"{level}_candidates.pdf")


def correlations(descriptors, output):
    fig = plt.figure(figsize=(5.5, 3.0))
    positions = ([0.1, 0.25, 0.335, 0.64], [0.57, 0.25, 0.335, 0.64])
    for level, position, title in zip(
        ("graph", "node"), positions, ("(a) Graph candidates", "(b) Node candidates")
    ):
        prefix = "graph_" if level == "graph" else "node_mean_"
        raw = descriptors[[f"{prefix}{k}" for k in range(5)]].copy()
        raw.columns = FEATURES[level]
        ordered = raw.corr(method="pearson")
        ordered.rename_axis("feature").to_csv(
            output / "data" / f"{level}_candidate_correlation.csv"
        )
        matrix = ordered.to_numpy(float)
        if matrix.shape != (5, 5) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {level} correlation matrix")
        ax = fig.add_axes(position)
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(title, pad=7)
        ax.set_xticks(
            range(5), SHORT_FEATURES[level], rotation=45, ha="right", fontsize=7
        )
        ax.set_yticks(range(5), SHORT_FEATURES[level], fontsize=7)
        ax.tick_params(length=0, pad=3)
        for row in range(5):
            for col in range(5):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) >= 0.6 else "#202020",
                )
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
    cax = fig.add_axes([0.93, 0.285, 0.017, 0.56])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    fig.text(
        0.5,
        0.025,
        "Pearson correlation; node candidates use one mean per graph",
        ha="center",
        fontsize=7,
    )
    save(fig, output, "correlation_candidates.pdf")


def build(prepared, output):
    prepared, output = (prepared.resolve(), output.resolve())
    verify_prepared(prepared)
    path = prepared / "dataset_descriptors.csv"
    descriptors = pd.read_csv(path, dtype={"graph_id": str})
    required = [
        "graph_id",
        "label",
        *[f"graph_{k}" for k in range(5)],
        *[f"node_mean_{k}" for k in range(5)],
    ]
    if (
        not set(required).issubset(descriptors)
        or descriptors.graph_id.isna().any()
        or descriptors.graph_id.duplicated().any()
    ):
        raise ValueError(
            "Expected unique graph IDs and all five node-mean/graph descriptor candidates"
        )
    if len(descriptors) != 2000 or descriptors.label.value_counts().to_dict() != {
        k: 400 for k in range(5)
    }:
        raise ValueError("Expected 400 graphs in each of the five families")
    if not np.isfinite(descriptors[required[2:]].to_numpy(float)).all():
        raise ValueError("Raw descriptors must be finite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    configure()
    for level in ("graph", "node"):
        candidates(descriptors, level, output)
    correlations(descriptors, output)
    selection_files = [
        prepared / f"seed_{seed}" / "selection.json" for seed in range(42, 47)
    ]
    selections = [json.loads(p.read_text()) for p in selection_files]
    rows, frequencies = ([], [])
    for level in ("node", "graph"):
        for index, name in enumerate(FEATURES[level]):
            count = sum(
                (
                    index in selection[level]["selected_indices"]
                    for selection in selections
                )
            )
            rows.append([level.capitalize(), name, str(count)])
            frequencies.append(
                dict(
                    level=level,
                    feature_index=index,
                    feature=name,
                    selected_seed_count=count,
                    n_seeds=5,
                )
            )
    (output / "tables").mkdir(exist_ok=True)
    latex_table(
        output / "tables/features_retained.tex",
        "Candidate descriptors and the number of training partitions in which each was selected.",
        "tab:features_retained",
        ["Level", "Candidate", "Partitions"],
        rows,
        alignment="llr",
    )
    pd.DataFrame(frequencies).to_csv(
        output / "data/feature_selection_frequency.csv", index=False
    )
    summary = []
    for level in ("graph", "node"):
        prefix = "graph_" if level == "graph" else "node_mean_"
        for k, name in enumerate(FEATURES[level]):
            for label, family in enumerate(CLASSES):
                a = descriptors.loc[
                    descriptors.label == label, f"{prefix}{k}"
                ].to_numpy(float)
                summary.append(
                    dict(
                        level=level,
                        feature=name,
                        class_label=label,
                        class_name=family,
                        n_graphs=len(a),
                        mean=a.mean(),
                        sample_sd=a.std(ddof=1),
                        minimum=a.min(),
                        q25=np.quantile(a, 0.25),
                        median=np.median(a),
                        q75=np.quantile(a, 0.75),
                        maximum=a.max(),
                    )
                )
    pd.DataFrame(summary).to_csv(
        output / "data/candidate_descriptives.csv", index=False
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "graphs": len(descriptors),
                "figures": 3,
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot graph and node descriptors and summarise training-partition feature selection."
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.prepared, args.output)
