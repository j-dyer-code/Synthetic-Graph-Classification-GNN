import itertools

ARCH_SYN = ["GCN", "GAT", "GATV2", "SAGE", "GIN", "GTN"]
ARCH_REAL = ["GIN", "SAGE", "GTN"]
REAL = ["NCI1", "PROTEINS", "IMDB-BINARY"]
CONFIGS = [
    dict(hidden=h, dropout=d, lr=lr, weight_decay=0.0001, batch_size=16)
    for h, d, lr in itertools.product([64, 128], [0.0, 0.3], [0.001, 0.003])
]
SETTINGS = dict(
    version="matched-controls-v1",
    tuning_epochs=120,
    tuning_min_epochs=30,
    tuning_patience=20,
    final_epochs=240,
    final_min_epochs=40,
    final_patience=30,
    initializations=[0, 1, 2],
    selection="minimum calibrated validation cross entropy",
    primary_metric="balanced_accuracy",
    secondary_metrics=["accuracy", "macro_f1", "roc_auc"],
    precision="float32; TF32 disabled; deterministic algorithms",
    synthetic_splits="saved repeated 70/15/15 partitions (seeds 42..46)",
    real_splits="5 outer stratified group folds; first of 5 inner group folds for validation",
    duplicate_groups="five-round attributed WL digest; conservative grouping, not exact isomorphism",
    paired_selection="minimum mean validation loss across full/self arms at identical config",
    configs=CONFIGS,
)


def models(dataset):
    out = []
    regimes = ["enhanced"] if dataset == "synthetic" else ["native", "enhanced"]
    for regime in regimes:
        for arch in ARCH_SYN if dataset == "synthetic" else ARCH_REAL:
            for edge in ["full", "self"]:
                out.append(dict(arch=arch, edge=edge, regime=regime))
        out.append(dict(arch="DeepSets", edge="none", regime=regime))
    out.append(dict(arch="GraphMLP", edge="none", regime="enhanced"))
    return out


def key(model):
    return "-".join((model[x] for x in ["arch", "edge", "regime"]))


def tasks(datasets=("synthetic", *REAL), smoke=False):
    out = []
    for ds in datasets:
        for fold in range(1 if smoke else 5):
            for model in models(ds):
                base = f"{ds}/fold_{fold}/{key(model)}"
                trials = []
                for ci in range(1 if smoke else len(CONFIGS)):
                    tid = f"{base}/trial_{ci}"
                    out.append(
                        dict(
                            id=tid,
                            dataset=ds,
                            fold=fold,
                            model=model,
                            kind="trial",
                            config=ci,
                            deps=[],
                        )
                    )
                    trials.append(tid)
                for mode in ["independent"] + (
                    ["paired"] if model["edge"] in ("full", "self") else []
                ):
                    deps = list(trials)
                    if mode == "paired":
                        other = dict(
                            model, edge="self" if model["edge"] == "full" else "full"
                        )
                        deps += [
                            f"{ds}/fold_{fold}/{key(other)}/trial_{ci}"
                            for ci in range(1 if smoke else len(CONFIGS))
                        ]
                    for seed in range(1 if smoke else 3):
                        final_deps = (
                            deps + [f"{base}/independent_{seed}"]
                            if mode == "paired"
                            else deps
                        )
                        out.append(
                            dict(
                                id=f"{base}/{mode}_{seed}",
                                dataset=ds,
                                fold=fold,
                                model=model,
                                kind="final",
                                mode=mode,
                                initialization=seed,
                                deps=final_deps,
                            )
                        )
            for regime in ["enhanced"] if ds == "synthetic" else ["native", "enhanced"]:
                out.append(
                    dict(
                        id=f"{ds}/fold_{fold}/SVM-{regime}",
                        dataset=ds,
                        fold=fold,
                        kind="svm",
                        regime=regime,
                        deps=[],
                    )
                )
    return out
