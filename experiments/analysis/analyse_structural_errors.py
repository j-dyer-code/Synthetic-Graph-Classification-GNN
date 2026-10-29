import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from result_io import verify_prepared, main_run_path, verified_sources

SEEDS = (42, 43, 44, 45, 46)
MODELS = (
    "GCN",
    "GAT",
    "GATV2",
    "GIN",
    "SAGE",
    "GTN",
    "SVM",
    "Descriptor_SVM_tuned",
    "Graph2Vec_SVM",
)
CLASSES = {0: "ER", 1: "BA", 2: "WS", 3: "SBM", 4: "HK"}
FEATURES = {
    "graph_0": "degree_variance",
    "graph_1": "assortativity",
    "graph_2": "density",
    "graph_3": "global_transitivity",
    "graph_4": "average_path_length",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = [v for v in values if v is not None]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, measures):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple((row[k] for k in keys))].append(row)
    result = []
    for key, group in groups.items():
        row = dict(zip(keys, key))
        for measure in measures:
            row.update(
                {
                    f"{measure}_{name}": value
                    for name, value in stats([v[measure] for v in group]).items()
                }
            )
        result.append(row)
    return result


def analyse(results, baselines, prepared, output):
    results, baselines, prepared = (
        results.resolve(),
        baselines.resolve(),
        prepared.resolve(),
    )
    output = output.resolve()
    verify_prepared(prepared)

    def source(path):
        require(path.is_file(), f"Required source missing: {path}")
        for root in (prepared, results, baselines):
            if path.is_relative_to(root):
                return path
        raise ValueError(f"Source outside supplied directories: {path}")

    def read_csv(relative):
        with source(relative).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    raw = read_csv(prepared / "dataset_descriptors.csv")
    require(len(raw) == 2000, "Expected 2,000 descriptor rows")
    descriptors = {}
    for row in raw:
        graph_id = row["graph_id"]
        require(graph_id not in descriptors, f"Duplicate descriptor ID: {graph_id}")
        label = int(row["label"])
        values = {name: float(row[column]) for column, name in FEATURES.items()}
        require(
            all((math.isfinite(x) for x in values.values())),
            f"Nonfinite features: {graph_id}",
        )
        descriptors[graph_id] = {"label": label, **values}
    require(
        Counter((v["label"] for v in descriptors.values()))
        == Counter({c: 400 for c in CLASSES}),
        "Expected 400 graphs of each family",
    )
    confusions, profiles, error_shares = ([], [], [])
    test_ids_union = set()
    for seed in SEEDS:
        split_path = prepared / f"seed_{seed}" / "split.json"
        split = json.loads(source(split_path).read_text())
        require(split["seed"] == seed, f"Split seed mismatch: {seed}")
        groups = split["ids"]
        split_ids = {
            key: set(map(str, groups[key])) for key in ("train", "validation", "test")
        }
        for key, size in (("train", 1400), ("validation", 300), ("test", 300)):
            require(
                len(groups[key]) == len(split_ids[key]) == size,
                f"Invalid {key} split: {seed}",
            )
        require(
            not (
                split_ids["train"] & split_ids["validation"]
                or split_ids["train"] & split_ids["test"]
                or split_ids["validation"] & split_ids["test"]
            ),
            f"Split overlap: {seed}",
        )
        require(
            set.union(*split_ids.values()) == set(descriptors),
            f"Split corpus mismatch: {seed}",
        )
        test_ids_union.update(split_ids["test"])
        selection_path = prepared / f"seed_{seed}" / "selection.json"
        selection = json.loads(source(selection_path).read_text())
        retained = {
            list(FEATURES.values())[index]
            for index in selection["graph"]["selected_indices"]
        }
        for model in MODELS:
            run = main_run_path(results, baselines, model, seed)
            prediction_path = run / "test_predictions.csv"
            for path in verified_sources(run, [prediction_path]):
                source(path)
            raw_predictions = read_csv(prediction_path)
            require(
                len(raw_predictions) == 300, f"Expected 300 predictions: {model}/{seed}"
            )
            prediction_ids = [row["graph_id"] for row in raw_predictions]
            require(
                len(set(prediction_ids)) == 300
                and set(prediction_ids) == split_ids["test"],
                f"Predictions do not match paired test split: {model}/{seed}",
            )
            predictions = []
            for row in raw_predictions:
                graph_id = row["graph_id"]
                true, predicted = (int(row["true_label"]), int(row["predicted_label"]))
                require(
                    true == descriptors[graph_id]["label"] and predicted in CLASSES,
                    f"Prediction label mismatch: {model}/{seed}/{graph_id}",
                )
                predictions.append(
                    {"graph_id": graph_id, "true": true, "predicted": predicted}
                )
            require(
                Counter((p["true"] for p in predictions))
                == Counter({c: 60 for c in CLASSES}),
                f"Unbalanced held-out labels: {model}/{seed}",
            )
            counts = Counter(((p["true"], p["predicted"]) for p in predictions))
            total_errors = sum((p["true"] != p["predicted"] for p in predictions))
            er_sbm_errors = counts[0, 3] + counts[3, 0]
            error_shares.append(
                {
                    "model": model,
                    "seed": seed,
                    "test_count": 300,
                    "error_count": total_errors,
                    "er_sbm_error_count": er_sbm_errors,
                    "er_sbm_share_of_errors_percent": 100 * er_sbm_errors / total_errors
                    if total_errors
                    else None,
                }
            )
            for true in CLASSES:
                for predicted in CLASSES:
                    confusions.append(
                        {
                            "model": model,
                            "seed": seed,
                            "true_class": CLASSES[true],
                            "predicted_class": CLASSES[predicted],
                            "count": counts[true, predicted],
                            "row_percent": 100 * counts[true, predicted] / 60,
                        }
                    )
            for true, wrong in ((0, 3), (3, 0)):
                errors = [
                    p
                    for p in predictions
                    if p["true"] == true and p["predicted"] == wrong
                ]
                correct = [
                    p
                    for p in predictions
                    if p["true"] == true and p["predicted"] == true
                ]
                for feature in FEATURES.values():
                    error_values = [descriptors[p["graph_id"]][feature] for p in errors]
                    correct_values = [
                        descriptors[p["graph_id"]][feature] for p in correct
                    ]
                    error_mean = (
                        statistics.fmean(error_values) if error_values else None
                    )
                    correct_mean = (
                        statistics.fmean(correct_values) if correct_values else None
                    )
                    difference = (
                        error_mean - correct_mean if errors and correct else None
                    )
                    profiles.append(
                        {
                            "model": model,
                            "seed": seed,
                            "true_class": CLASSES[true],
                            "error_class": CLASSES[wrong],
                            "feature": feature,
                            "retained_graph_channel_in_gnns_and_descriptor_svms": feature
                            in retained,
                            "error_count": len(errors),
                            "correct_count": len(correct),
                            "error_group_below_five": len(errors) < 5,
                            "correct_group_below_five": len(correct) < 5,
                            "error_mean": error_mean,
                            "correct_mean": correct_mean,
                            "error_minus_correct": difference,
                        }
                    )
    profile_keys = ["model", "true_class", "error_class", "feature"]
    profile_summary = summarize(
        profiles,
        profile_keys,
        [
            "error_count",
            "correct_count",
            "error_mean",
            "correct_mean",
            "error_minus_correct",
        ],
    )
    for row in profile_summary:
        group = [p for p in profiles if all((p[k] == row[k] for k in profile_keys))]
        delta = [
            p["error_minus_correct"]
            for p in group
            if p["error_minus_correct"] is not None
        ]
        retained_count = sum(
            (p["retained_graph_channel_in_gnns_and_descriptor_svms"] for p in group)
        )
        row.update(
            {
                "retained_graph_channel_repetitions": retained_count,
                "retained_graph_channel_in_all_repetitions": retained_count
                == len(SEEDS),
                "negative_delta_repetitions": sum((v < 0 for v in delta)),
                "zero_delta_repetitions": sum((v == 0 for v in delta)),
                "positive_delta_repetitions": sum((v > 0 for v in delta)),
                "missing_delta_repetitions": len(SEEDS) - len(delta),
                "error_count_min": min((p["error_count"] for p in group)),
                "error_count_max": max((p["error_count"] for p in group)),
                "error_group_below_five_repetitions": sum(
                    (p["error_group_below_five"] for p in group)
                ),
                "correct_group_below_five_repetitions": sum(
                    (p["correct_group_below_five"] for p in group)
                ),
            }
        )
    tables = {
        "confusion_per_repetition.csv": confusions,
        "confusion_summary.csv": summarize(
            confusions,
            ["model", "true_class", "predicted_class"],
            ["count", "row_percent"],
        ),
        "error_share_per_repetition.csv": error_shares,
        "error_share_summary.csv": summarize(
            error_shares,
            ["model"],
            ["error_count", "er_sbm_error_count", "er_sbm_share_of_errors_percent"],
        ),
        "descriptor_profiles_per_repetition.csv": profiles,
        "descriptor_profiles_summary.csv": profile_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    shared = [p for p in profiles if p["model"] != "Graph2Vec_SVM"]
    direction_checks = []
    for true_class, feature, sign in (
        ("SBM", "degree_variance", -1),
        ("SBM", "assortativity", -1),
        ("SBM", "global_transitivity", -1),
        ("ER", "assortativity", 1),
        ("ER", "global_transitivity", 1),
    ):
        values = [
            p["error_minus_correct"]
            for p in shared
            if p["true_class"] == true_class and p["feature"] == feature
        ]
        direction_checks.append(
            {
                "true_class": true_class,
                "feature": feature,
                "error_mean_relative_to_correct": "lower" if sign == -1 else "higher",
                "matching_classifier_repetitions": sum(
                    (v is not None and v * sign > 0 for v in values)
                ),
                "classifier_repetitions": len(values),
            }
        )
    report = {
        "analysis": "Descriptive post hoc structural profiles of held-out classification errors",
        "status": "complete",
        "validated_runs": 45,
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "graphs_in_corpus": len(descriptors),
        "test_appearances_per_model": 1500,
        "distinct_test_graphs_per_model": len(test_ids_union),
        "summary_unit": "Equal-weight repetition-level statistics; sample SD across five repetitions",
        "scope": [
            "All five raw graph-descriptor candidates are reported without fitting a new model or changing predictions.",
            "Each directional error group is compared with correctly classified graphs of the same true family in the same test split.",
            "The retained-channel flag describes GNN and descriptor-SVM inputs; Graph2Vec does not consume these handcrafted channels.",
            "Consistency counts report the number of classifier/evaluation combinations with the indicated direction.",
            "Groups below five graphs are flagged and retained; empty groups produce missing statistics.",
        ],
        "direction_consistency_shared_descriptor_classifiers": direction_checks,
        "output_rows": {name: len(rows) for name, rows in tables.items()},
    }
    (output / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "validated_runs": report["validated_runs"],
                "distinct_test_graphs_per_model": len(test_ids_union),
                "output_rows": report["output_rows"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarise held-out classification errors and their structural descriptor profiles."
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Benchmark output directory"
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        required=True,
        help="Tuned-SVM and Graph2Vec output directory",
    )
    parser.add_argument(
        "--prepared", type=Path, help="Prepared inputs (default: RESULTS/prepared)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyse(
        args.results,
        args.baselines,
        args.prepared or args.results / "prepared",
        args.output,
    )


if __name__ == "__main__":
    main()
