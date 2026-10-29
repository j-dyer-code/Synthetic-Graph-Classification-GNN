import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from common import atomic_json
from metrics import classification_metrics, integer_labels
from sklearn.metrics import log_loss

CLASSES = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "holme_kim",
]
C_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMA_MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)
REPRESENTATIONS = ("Graph2Vec_SVM", "Descriptor_SVM_tuned")


def fit_classifier(X, y, graph_ids, split, seed, output, representation, identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown representation")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("identity must be a config digest")
    graph_ids = [str(value) for value in graph_ids]
    split = {key: [str(value) for value in values] for key, values in split.items()}
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions")
    flattened = [
        value for part in ("train", "validation", "test") for value in split[part]
    ]
    if (
        len(set(graph_ids)) != len(graph_ids)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(graph_ids)
    ):
        raise ValueError("Duplicate, overlapping or missing graph IDs")
    if len(np.shape(X)) != 2 or np.shape(X)[0] != len(graph_ids) or np.shape(X)[1] < 1:
        raise ValueError("Representation dimensions disagree with graph IDs")
    if np.shape(y) != (len(graph_ids),):
        raise ValueError("Label dimensions disagree with graph IDs")
    mapping = {value: i for i, value in enumerate(graph_ids)}
    index = {
        part: np.asarray([mapping[value] for value in values], dtype=int)
        for part, values in split.items()
    }
    train = np.asarray(X[index["train"]], dtype=np.float64)
    validation = np.asarray(X[index["validation"]], dtype=np.float64)
    y_train = integer_labels(y[index["train"]], len(CLASSES))
    y_validation = integer_labels(y[index["validation"]], len(CLASSES))
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Non-finite training or validation features")
    if set(y_train) != set(range(5)) or set(y_validation) != set(range(5)):
        raise ValueError("All five classes must occur in training and validation")
    variance = float(train.var())
    gamma_base = 1.0 / (train.shape[1] * variance) if variance > 0 else 1.0
    rows = []
    started = time.perf_counter()
    for c in C_VALUES:
        for multiplier in GAMMA_MULTIPLIERS:
            gamma = float(gamma_base * multiplier)
            model = SVC(
                C=c,
                gamma=gamma,
                kernel="rbf",
                probability=False,
                decision_function_shape="ovr",
                cache_size=1024,
                random_state=int(seed),
            )
            trial_started = time.perf_counter()
            model.fit(train, y_train)
            predicted = model.predict(validation)
            rows.append(
                {
                    "candidate": len(rows),
                    "C": c,
                    "gamma_multiplier": multiplier,
                    "gamma": gamma,
                    "validation_accuracy": float(np.mean(predicted == y_validation)),
                    "fit_and_validation_seconds": float(
                        time.perf_counter() - trial_started
                    ),
                }
            )
    selected = max(range(len(rows)), key=lambda i: rows[i]["validation_accuracy"])
    tuning_seconds = float(time.perf_counter() - started)
    best = rows[selected]
    parameters = {
        "fingerprint": identity,
        "seed": int(seed),
        "representation": representation,
        "classes": CLASSES,
        "split_ids": split,
        "kernel": "rbf",
        "probability": True,
        "C": best["C"],
        "gamma": best["gamma"],
        "gamma_multiplier": best["gamma_multiplier"],
        "gamma_base": gamma_base,
        "training_flattened_variance": variance,
        "feature_count": int(train.shape[1]),
        "selected_candidate": selected,
        "selected_validation_accuracy": best["validation_accuracy"],
        "selection_metric": "validation_accuracy",
        "tie_break": "first maximum; C ascending then gamma_multiplier ascending",
        "calibration_random_state": int(seed),
        "refit_training_and_validation": False,
    }
    atomic_json(
        output / "grid.json",
        {
            "fingerprint": identity,
            "purpose": "validation_selection",
            "C_values": list(C_VALUES),
            "gamma_multipliers": list(GAMMA_MULTIPLIERS),
            "gamma_base": gamma_base,
            "candidates": rows,
            "selected_candidate": selected,
            "selection_metric": "validation_accuracy",
            "test_used_for_selection": False,
        },
    )
    atomic_json(output / "parameters.json", parameters)
    model = SVC(
        C=best["C"],
        gamma=best["gamma"],
        kernel="rbf",
        probability=True,
        decision_function_shape="ovr",
        cache_size=1024,
        random_state=int(seed),
    )
    started = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = float(time.perf_counter() - started)
    if not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError("Fitted SVM class order differs")
    temporary = output / "svm.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, output / "svm.joblib")
    test = np.asarray(X[index["test"]], dtype=np.float64)
    y_test = integer_labels(y[index["test"]], len(CLASSES))
    if not np.isfinite(test).all() or set(y_test) != set(range(5)):
        raise ValueError("Invalid held-out features or labels")
    started = time.perf_counter()
    predicted = model.predict(test)
    probability = model.predict_proba(test)
    decision = model.decision_function(test)
    inference_seconds = float(time.perf_counter() - started)
    frame = pd.DataFrame(
        {
            "graph_id": split["test"],
            "true_label": y_test,
            "predicted_label": predicted,
            "probability_argmax": probability.argmax(axis=1),
        }
    )
    for label in range(5):
        frame[f"probability_{label}"] = probability[:, label]
    for label in range(5):
        frame[f"decision_{label}"] = decision[:, label]
    temporary = output / "test_predictions.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output / "test_predictions.csv")
    test_metrics = classification_metrics(y_test, predicted, probability, CLASSES)
    normalized = probability.astype(np.float64)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)
    test_metrics["probability_log_loss"] = float(
        log_loss(y_test, normalized, labels=np.arange(5))
    )
    metrics = {
        "fingerprint": identity,
        "seed": int(seed),
        "representation": representation,
        "classes": CLASSES,
        "test": test_metrics,
        "timings": {
            "tuning_seconds": tuning_seconds,
            "svm_fit_seconds": fit_seconds,
            "test_inference_seconds": inference_seconds,
        },
        "timing_scopes": {
            "tuning_seconds": "25 sequential SVC fits and validation predictions; excludes representation construction and serialization",
            "svm_fit_seconds": "Selected SVC.fit on training rows, including internal probability calibration",
            "test_inference_seconds": "SVC.predict, predict_proba and decision_function on held-out rows; excludes representation inference and serialization",
        },
    }
    atomic_json(output / "metrics.json", metrics)
    return metrics
