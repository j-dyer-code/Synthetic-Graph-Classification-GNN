import argparse
import collections
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from core import read, write, done_valid, verify_run, sha, digest
from protocol import ARCH_SYN, ARCH_REAL, REAL, key


def csvwrite(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def contrast_statistics(differences, ratios):
    differences = np.asarray(differences, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if differences.shape != (5,) or ratios.shape != (5,):
        raise ValueError(
            "Contrasts require five fold differences and test-to-training ratios"
        )
    if (
        not np.isfinite(differences).all()
        or not np.isfinite(ratios).all()
        or (ratios <= 0).any()
    ):
        raise ValueError(
            "Contrast inputs must be finite, with positive test-to-training ratios"
        )
    mean = float(np.mean(differences))
    var = float(np.var(differences, ddof=1))
    degenerate = math.sqrt(var) <= 32 * np.finfo(float).eps * max(
        1.0, float(np.abs(differences).max())
    )
    se = math.sqrt((1 / 5 + float(np.mean(ratios))) * var)
    critical = float(student_t.ppf(1 - 0.05 / (2 * 24), 4))
    p = None if degenerate else float(2 * student_t.sf(abs(mean / se), 4))
    return dict(
        mean_difference_pp=100 * mean,
        corrected_se_pp=100 * se,
        bonferroni_95_lower_pp=None if degenerate else 100 * (mean - critical * se),
        bonferroni_95_upper_pp=None if degenerate else 100 * (mean + critical * se),
        uncorrected_p=p,
        degenerate_variance=degenerate,
        fold_differences_pp=json.dumps([100 * x for x in differences]),
        method="approximate corrected resampled t; fold-level means over 3 initializations; df=4",
    )


def adjust_contrast_p_values(contrasts):
    if len(contrasts) != 24:
        raise ValueError("Holm adjustment requires the 24 prespecified contrasts")
    for row in contrasts:
        row["holm_adjusted_p"] = None
    order = sorted(
        (i for i, row in enumerate(contrasts) if row["uncorrected_p"] is not None),
        key=lambda i: contrasts[i]["uncorrected_p"],
    )
    last = 0
    for rank, i in enumerate(order):
        p = contrasts[i]["uncorrected_p"]
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(
                "Estimated contrast p-values must be finite and lie in [0, 1]"
            )
        last = max(last, min(1.0, p * (24 - rank)))
        contrasts[i]["holm_adjusted_p"] = last


def checked_predictions(folder, phase, task, cfg, records, saved):
    with np.load(folder / (phase + ".npz"), allow_pickle=False) as arrays:
        ids = arrays["indices"]
        labels = arrays["labels"]
        if ids.tolist() != cfg["split"][phase]:
            raise ValueError("Wrong " + phase + " membership: " + task["id"])
        if labels.tolist() != [records[int(i)]["label"] for i in ids]:
            raise ValueError("Wrong " + phase + " labels")
        if task["kind"] == "svm":
            pred = arrays["predicted"]
            scores = arrays["decision"]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite SVM scores")
            probability = None
        else:
            probability = arrays["probabilities"]
            logits = arrays["logits"]
            if not np.isfinite(logits).all() or not np.isfinite(probability).all():
                raise ValueError("Nonfinite predictions")
            expected = np.exp(logits - logits.max(1, keepdims=True))
            expected /= expected.sum(1, keepdims=True)
            if not np.allclose(expected, probability, atol=1e-06, rtol=1e-06):
                raise ValueError("Probabilities do not match logits")
            pred = probability.argmax(1)
        values = dict(
            accuracy=float(accuracy_score(labels, pred)),
            balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
            macro_f1=float(f1_score(labels, pred, average="macro", zero_division=0)),
        )
        if probability is not None:
            values["roc_auc"] = float(
                roc_auc_score(labels, probability[:, 1])
                if probability.shape[1] == 2
                else roc_auc_score(
                    labels, probability, multi_class="ovr", average="macro"
                )
            )
            values["loss"] = float(
                log_loss(labels, probability, labels=list(range(probability.shape[1])))
            )
        elif scores.ndim == 1:
            values["roc_auc"] = float(roc_auc_score(labels, scores))
        for name, value in values.items():
            if not np.isclose(value, saved[name], atol=1e-10, rtol=1e-10):
                raise ValueError("Recomputed " + phase + " metric mismatch: " + name)
    return {name: value for name, value in values.items() if name != "loss"}


def analyse(run, smoke=False):
    run = Path(run)
    manifest = verify_run(run)
    if bool(manifest.get("smoke", False)) != smoke:
        raise ValueError("Smoke mode must agree with the prepared run")
    ts = read(run / "tasks.json")
    rows = []
    group = collections.defaultdict(list)
    limits = []
    for task in ts:
        folder = run / "tasks" / task["id"]
        if not done_valid(folder, True):
            raise RuntimeError("Incomplete task: " + task["id"])
        met = read(folder / "metrics.json")
        if met["task"] != task:
            raise ValueError("Task identity mismatch: " + task["id"])
        cfg = read(run / "data" / task["dataset"] / f"fold_{task['fold']}.json")
        records = read(run / "data" / task["dataset"] / "records.json")
        checked_predictions(folder, "validation", task, cfg, records, met["validation"])
        if task["kind"] == "svm":
            frozen = read(folder / "selection_frozen.json")
            history = read(folder / "history.json")
            if (
                frozen["training_indices"] != cfg["split"]["train"]
                or frozen["validation_indices"] != cfg["split"]["validation"]
            ):
                raise ValueError("SVM selection membership differs")
            winner = max(history, key=lambda row: row["validation_balanced_accuracy"])
            selected_params = {key: winner[key] for key in ("C", "gamma_multiplier")}
            if met["params"] != selected_params or frozen["params"] != selected_params:
                raise ValueError("SVM validation selection differs")
        if task["kind"] != "svm":
            import torch
            from normalization import tensor_mapping_hash
            from learning import choose, seed_value
            from protocol import CONFIGS

            weights = torch.load(
                folder / "best.pt", map_location="cpu", weights_only=True
            )
            actual = tensor_mapping_hash(weights.items())
            calibration = read(folder / "calibration.json")
            if (
                actual != met["checkpoint_state_sha256"]
                or actual != calibration["calibrated_state_sha256"]
            ):
                raise ValueError("Checkpoint/calibration state mismatch")
            ci = choose(task, run)
            if met["config_index"] != ci or met["params"] != CONFIGS[ci]:
                raise ValueError("Validation configuration selection mismatch")
            phase = 0 if task["kind"] == "trial" else 1 + task["initialization"]
            if met["seed"] != seed_value(
                task["dataset"], task["fold"], task["model"]["arch"], ci, phase
            ):
                raise ValueError("Initialization seed differs")
            expected_identity = digest(
                dict(
                    task=task,
                    params=met["params"],
                    data=cfg,
                    source=manifest["source_hashes"],
                )
            )
            if met["identity"] != expected_identity:
                raise ValueError("Training identity differs")
            if calibration["training_ids"] != cfg["training_ids"]:
                raise ValueError("Calibration training IDs differ")
            if task["kind"] == "final":
                selected = read(folder / "selection_frozen.json")
                if (
                    selected["config_index"] != ci
                    or selected["state_sha256"] != actual
                    or selected["selected_epoch"] != met["selected_epoch"]
                    or (selected["dependencies"] != task["deps"])
                ):
                    raise ValueError("Frozen final selection differs")
                if met.get("reused_from"):
                    source = run / "tasks" / met["reused_from"]
                    if not done_valid(source, True):
                        raise ValueError("Reuse source is incomplete")
                    original = read(source / "metrics.json")
                    if (
                        met["reused_source_identity"] != original["identity"]
                        or original["config_index"] != ci
                        or original["seed"] != met["seed"]
                    ):
                        raise ValueError("Reuse identity differs")
                    for name in [
                        "best.pt",
                        "calibration.json",
                        "history.json",
                        "validation.npz",
                        "test.npz",
                    ]:
                        if sha(source / name) != sha(folder / name):
                            raise ValueError("Reused artifact differs: " + name)
        if task["kind"] == "trial":
            if (folder / "test.npz").exists():
                raise ValueError("Trial contains prohibited test outputs")
            if met["hit_epoch_cap"]:
                limits.append(task["id"])
            continue
        recomputed = checked_predictions(
            folder, "test", task, cfg, records, met["test"]
        )
        model = "SVM-" + task["regime"] if task["kind"] == "svm" else key(task["model"])
        mode = task.get("mode", "independent")
        fold = task["fold"]
        ds = task["dataset"]
        row = dict(
            dataset=ds,
            fold=fold,
            model=model,
            selection=mode,
            initialization=task.get("initialization", 0),
            **recomputed,
            seconds=met["seconds"],
            reused=bool(met.get("reused_from")),
        )
        rows.append(row)
        group[ds, model, mode, fold].append(row)
    folded = []
    lookup = {}
    for (ds, model, mode, fold), rs in sorted(group.items()):
        record = dict(dataset=ds, model=model, selection=mode, fold=fold)
        for metric in ["balanced_accuracy", "accuracy", "macro_f1", "roc_auc"]:
            vals = [r[metric] for r in rs if metric in r]
            if vals:
                record[metric] = float(np.mean(vals))
        folded.append(record)
        lookup[ds, model, mode, fold] = record["balanced_accuracy"]
    aggregates = collections.defaultdict(list)
    for r in folded:
        aggregates[r["dataset"], r["model"], r["selection"]].append(r)
    summary = []
    for (ds, model, mode), rs in sorted(aggregates.items()):
        r = dict(dataset=ds, model=model, selection=mode, folds=len(rs))
        for metric in ["balanced_accuracy", "accuracy", "macro_f1", "roc_auc"]:
            v = [x[metric] for x in rs if metric in x]
            if v:
                r[metric + "_mean"] = float(np.mean(v))
                r[metric + "_fold_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary.append(r)
    contrasts = []

    def contrast(ds, arch, kind, a, b, mode):
        differences = []
        ratios = []
        for fold in range(5):
            differences.append(lookup[ds, a, mode, fold] - lookup[ds, b, mode, fold])
            split = read(run / "data" / ds / f"fold_{fold}.json")["split"]
            ratios.append(len(split["test"]) / len(split["train"]))
        contrasts.append(
            dict(
                dataset=ds,
                architecture=arch,
                contrast=kind,
                selection=mode,
                **contrast_statistics(differences, ratios),
            )
        )

    if not smoke:
        for ds in ["synthetic", *REAL]:
            for arch in ARCH_SYN if ds == "synthetic" else ARCH_REAL:
                contrast(
                    ds,
                    arch,
                    "full_minus_self_enhanced",
                    f"{arch}-full-enhanced",
                    f"{arch}-self-enhanced",
                    "paired",
                )
                if ds != "synthetic":
                    contrast(
                        ds,
                        arch,
                        "enhanced_minus_native_full",
                        f"{arch}-full-enhanced",
                        f"{arch}-full-native",
                        "independent",
                    )
        adjust_contrast_p_values(contrasts)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    csvwrite(out / "per_fit.csv", rows)
    csvwrite(out / "fold_means.csv", folded)
    csvwrite(out / "summary.csv", summary)
    csvwrite(out / "primary_comparisons.csv", contrasts)
    write(
        out / "audit.json",
        dict(
            status="passed",
            purpose="software_test" if smoke else "research",
            synthetic_scope=manifest.get("synthetic_scope"),
            tasks=len(ts),
            heldout_fits=len(rows),
            all_saved_metrics_recomputed=True,
            tuning_hits_epoch_cap=limits,
            statistics="24 prespecified corrected-resampling contrasts; five fold means; three neural initialisations per fold",
            scope="Synthetic generator classification and grouped TU graph classification",
        ),
    )
    print("Analysis and integrity checks passed:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Validate matched-control predictions and summarise prespecified comparisons."
    )
    p.add_argument("--run", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    analyse(a.run, a.smoke)
