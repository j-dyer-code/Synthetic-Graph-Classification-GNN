from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from publication_style import MODELS

LABELS = tuple(range(5))
SEEDS = tuple(range(42, 47))
GRID = np.linspace(0.0, 1.0, 201)


def paired_accuracy_intervals(runs):
    mapping = {(r.architecture, r.seed): r.aggregate["accuracy"] for r in runs}
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if len(runs) != 45 or set(mapping) != expected:
        raise ValueError(
            "Paired comparisons require all 45 classifier/seed combinations"
        )
    rows, differences = ([], [])
    for a, b in combinations(MODELS, 2):
        delta = np.array([mapping[a, seed] - mapping[b, seed] for seed in SEEDS])
        if not np.isfinite(delta).all():
            raise ValueError("Paired accuracies must be finite")
        avg, sd = (delta.mean(), delta.std(ddof=1))
        degenerate = sd <= 32 * np.finfo(float).eps * max(1.0, np.abs(delta).max())
        for seed, d in zip(SEEDS, delta):
            differences.append(
                dict(
                    architecture_a=a,
                    architecture_b=b,
                    seed=seed,
                    accuracy_difference=d,
                    accuracy_difference_percentage_points=100 * d,
                )
            )
        for denominator in (1400, 1700):
            se = np.nan if degenerate else np.sqrt(1 / 5 + 300 / denominator) * sd
            point = t.ppf(0.975, 4) * se
            family = t.ppf(1 - 0.05 / (2 * 36), 4) * se
            rows.append(
                dict(
                    architecture_a=a,
                    architecture_b=b,
                    n_pairs=5,
                    degrees_of_freedom=4,
                    family_size=36,
                    n_test=300,
                    n_denominator=denominator,
                    interval_status="not_estimable_zero_or_roundoff_variance"
                    if degenerate
                    else "approximate",
                    accuracy_difference_mean=avg,
                    accuracy_difference_sd=sd,
                    corrected_standard_error=se,
                    pointwise95_low=avg - point,
                    pointwise95_high=avg + point,
                    bonferroni95_low=avg - family,
                    bonferroni95_high=avg + family,
                    bonferroni95_contains_zero=None
                    if degenerate
                    else bool(avg - family <= 0 <= avg + family),
                )
            )
    return (pd.DataFrame(rows), pd.DataFrame(differences))


class Run:
    def __init__(
        self,
        seed,
        architecture,
        path,
        config,
        metrics,
        tuning,
        parameters,
        history,
        predictions,
        aggregate,
        per_class,
        confusion,
        curves,
    ):
        self.seed = seed
        self.architecture = architecture
        self.path = path
        self.config = config
        self.metrics = metrics
        self.tuning = tuning
        self.parameters = parameters
        self.history = history
        self.predictions = predictions
        self.aggregate = aggregate
        self.per_class = per_class
        self.confusion = confusion
        self.curves = curves


def mean_sd(values):
    values = np.asarray(list(values), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A sample SD requires at least two finite observations")
    return (float(values.mean()), float(values.std(ddof=1)))


def classification_metrics(predictions):
    required = {"graph_id", "true_label", "predicted_label", "probability_argmax"}
    required.update((f"probability_{k}" for k in LABELS))
    required.update((f"decision_{k}" for k in LABELS))
    if not required.issubset(predictions.columns):
        raise ValueError(
            f"Missing prediction columns: {sorted(required - set(predictions.columns))}"
        )
    if (
        predictions.empty
        or predictions.graph_id.isna().any()
        or predictions.graph_id.duplicated().any()
    ):
        raise ValueError("Empty predictions or missing/duplicate graph IDs")
    for column in ("true_label", "predicted_label", "probability_argmax"):
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"Non-integer labels in {column}")
        if not set(values.astype(int)).issubset(LABELS):
            raise ValueError(f"Unknown labels in {column}")
    y = predictions.true_label.to_numpy(dtype=int)
    predicted = predictions.predicted_label.to_numpy(dtype=int)
    probability = predictions[[f"probability_{k}" for k in LABELS]].to_numpy(
        dtype=float
    )
    decisions = predictions[[f"decision_{k}" for k in LABELS]].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not np.isfinite(decisions).all():
        raise ValueError("Non-finite probabilities or decisions")
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError("Probabilities outside [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-06):
        raise ValueError("Probability rows do not sum to one")
    if not np.array_equal(
        probability.argmax(axis=1), predictions.probability_argmax.to_numpy(dtype=int)
    ):
        raise ValueError("Saved probability_argmax disagrees with probability columns")
    if set(y) != set(LABELS):
        raise ValueError("All five classes must be present for class-wise ROC and PR")
    p, r, f, support = precision_recall_fscore_support(
        y, predicted, labels=LABELS, average=None, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_precision": float(np.average(p, weights=support)),
        "weighted_recall": float(np.average(r, weights=support)),
        "weighted_f1": float(np.average(f, weights=support)),
        "probability_log_loss": float(
            log_loss(
                y, probability / probability.sum(axis=1, keepdims=True), labels=LABELS
            )
        ),
        "probability_argmax_disagreements": int(
            np.sum(predicted != probability.argmax(axis=1))
        ),
    }
    per_class, curves = ([], {})
    for k in LABELS:
        binary = (y == k).astype(int)
        auc = float(roc_auc_score(binary, probability[:, k]))
        ap = float(average_precision_score(binary, probability[:, k]))
        fpr, tpr, _ = roc_curve(binary, probability[:, k], drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(binary, probability[:, k])
        roc_y = np.interp(GRID, fpr, tpr)
        roc_y[0], roc_y[-1] = (0.0, 1.0)
        increasing_recall, increasing_precision = (recall[::-1], precision[::-1])
        pr_index = np.searchsorted(increasing_recall, GRID, side="left")
        pr_y = increasing_precision[np.minimum(pr_index, len(increasing_precision) - 1)]
        curves[str(k)] = {"roc": roc_y, "pr": pr_y}
        per_class.append(
            {
                "class_label": k,
                "precision": float(p[k]),
                "recall": float(r[k]),
                "f1": float(f[k]),
                "support": int(support[k]),
                "roc_auc_ovr": auc,
                "average_precision_ovr": ap,
            }
        )
    curves["macro"] = {
        kind: np.mean([curves[str(k)][kind] for k in LABELS], axis=0)
        for kind in ("roc", "pr")
    }
    result["roc_auc_macro_ovr"] = float(np.mean([v["roc_auc_ovr"] for v in per_class]))
    result["average_precision_macro_ovr"] = float(
        np.mean([v["average_precision_ovr"] for v in per_class])
    )
    result["roc_auc_weighted_ovr"] = float(
        np.average([v["roc_auc_ovr"] for v in per_class], weights=support)
    )
    result["average_precision_weighted_ovr"] = float(
        np.average([v["average_precision_ovr"] for v in per_class], weights=support)
    )
    return (result, per_class, confusion_matrix(y, predicted, labels=LABELS), curves)


def descriptive_summary(frame, group_columns, metrics):
    output = []
    for group, block in frame.groupby(group_columns, sort=False, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_columns, group))
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="raise")
            if values.isna().any():
                if values.notna().any():
                    raise ValueError(
                        f"Mixed missing/observed values for {group}: {metric}"
                    )
                row[metric + "_n"] = 0
                row[metric + "_mean"] = None
                row[metric + "_sd"] = None
            else:
                average, sd = mean_sd(values)
                row[metric + "_n"] = len(values)
                row[metric + "_mean"] = average
                row[metric + "_sd"] = sd
        output.append(row)
    return pd.DataFrame(output)


def collect_frames(runs, classes):
    aggregate, per_class, confusion_rows, curve_rows, history_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for run in runs:
        base = {"architecture": run.architecture, "seed": run.seed}
        aggregate.append({**base, **run.aggregate})
        for values in run.per_class:
            per_class.append(
                {**base, "class_name": classes[int(values["class_label"])], **values}
            )
        normalized = run.confusion / run.confusion.sum(axis=1, keepdims=True)
        for true in LABELS:
            for predicted in LABELS:
                confusion_rows.append(
                    {
                        **base,
                        "true_label": true,
                        "true_class": classes[true],
                        "predicted_label": predicted,
                        "predicted_class": classes[predicted],
                        "count": int(run.confusion[true, predicted]),
                        "row_normalized": float(normalized[true, predicted]),
                    }
                )
        for name, kinds in run.curves.items():
            for kind, values in kinds.items():
                for x, value in zip(GRID, values):
                    curve_rows.append(
                        {
                            **base,
                            "class_label": name,
                            "class_name": "macro"
                            if name == "macro"
                            else classes[int(name)],
                            "curve": kind,
                            "x": float(x),
                            "y": float(value),
                        }
                    )
        for epoch in run.history:
            history_rows.append({**base, **epoch})
    aggregate, per_class = (pd.DataFrame(aggregate), pd.DataFrame(per_class))
    confusion_rows, curve_rows = (
        pd.DataFrame(confusion_rows),
        pd.DataFrame(curve_rows),
    )
    frames = {
        "per_run_metrics": aggregate,
        "metrics_summary": descriptive_summary(
            aggregate, ["architecture"], list(runs[0].aggregate)
        ),
        "per_run_class_metrics": per_class,
        "class_metrics_summary": descriptive_summary(
            per_class,
            ["architecture", "class_label", "class_name"],
            [
                "precision",
                "recall",
                "f1",
                "support",
                "roc_auc_ovr",
                "average_precision_ovr",
            ],
        ),
        "per_run_confusion": confusion_rows,
        "confusion_summary": descriptive_summary(
            confusion_rows,
            [
                "architecture",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
            ],
            ["count", "row_normalized"],
        ),
        "per_run_display_curves": curve_rows,
        "display_curves_summary": descriptive_summary(
            curve_rows,
            ["architecture", "class_label", "class_name", "curve", "x"],
            ["y"],
        ),
        "histories": pd.DataFrame(history_rows),
    }
    history = frames["histories"]
    history_summary = []
    if not history.empty:
        for (architecture, epoch), values in history.groupby(
            ["architecture", "epoch"], sort=False
        ):
            entry = {
                "architecture": architecture,
                "epoch": int(epoch),
                "n_support": len(values),
            }
            for key in (
                "train_loss",
                "validation_loss",
                "train_accuracy",
                "validation_accuracy",
                "learning_rate",
            ):
                numbers = values[key].to_numpy(dtype=float)
                entry[key + "_mean"] = float(numbers.mean())
                entry[key + "_sd"] = (
                    float(numbers.std(ddof=1)) if len(numbers) > 1 else None
                )
            history_summary.append(entry)
    frames["history_summary"] = pd.DataFrame(history_summary)
    intervals, differences = paired_accuracy_intervals(runs)
    frames["paired_accuracy_intervals"] = intervals
    frames["paired_accuracy_differences"] = differences
    return frames
