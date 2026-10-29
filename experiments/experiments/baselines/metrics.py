import numpy as np


def integer_labels(values, class_count):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("Class labels must be a one-dimensional numeric array")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Class labels must contain finite integers")
    if ((values < 0) | (values >= class_count)).any():
        raise ValueError("Class labels are outside the declared class range")
    return values.astype(np.int64)


def classification_metrics(labels, predicted, probabilities, classes, loss=None):
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )

    labels = integer_labels(labels, len(classes))
    predicted = integer_labels(predicted, len(classes))
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(classes)
    if (
        labels.shape != predicted.shape
        or probabilities.shape != (len(labels), count)
        or (not len(labels))
    ):
        raise ValueError("Prediction dimensions differ")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (not np.allclose(probabilities.sum(1), 1, atol=1e-06))
    ):
        raise ValueError("Invalid predicted probabilities")
    if set(labels) != set(range(count)) or not set(predicted).issubset(range(count)):
        raise ValueError("All classes must occur in held-out labels")
    p, r, f, s = precision_recall_fscore_support(
        labels, predicted, labels=list(range(count)), zero_division=0
    )
    binary = np.eye(count)[labels]
    auc = [
        float(roc_auc_score(binary[:, j], probabilities[:, j])) for j in range(count)
    ]
    ap = [
        float(average_precision_score(binary[:, j], probabilities[:, j]))
        for j in range(count)
    ]
    return dict(
        accuracy=float(np.mean(labels == predicted)),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        macro_f1=float(f.mean()),
        weighted_f1=float(np.dot(f, s) / s.sum()),
        macro_roc_auc=float(np.mean(auc)),
        macro_average_precision=float(np.mean(ap)),
        loss=loss,
        confusion_matrix=confusion_matrix(
            labels, predicted, labels=list(range(count))
        ).tolist(),
        per_class=[
            dict(
                label=j,
                name=classes[j],
                precision=float(p[j]),
                recall=float(r[j]),
                f1=float(f[j]),
                support=int(s[j]),
                roc_auc=auc[j],
                average_precision=ap[j],
            )
            for j in range(count)
        ],
    )
