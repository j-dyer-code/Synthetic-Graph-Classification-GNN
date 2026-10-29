# GNNs Benchmarking for Synthetic Graph Classifications

This repository provides the experimental scripts for results reported in the paper: *Feature-Enhanced Graph Neural Networks for Classification of Synthetic Graph Generative Models: A Benchmarking Study*.

The results reported in the paper can be reproduced by following the instructions below. 

Run the commands below from this directory. Inputs and outputs are kept outside the source folders.

## Contents

| Directory | Experiments |
| --- | --- |
| `data/` | Synthetic graph generation and preparation |
| `benchmark/` | Six GNNs, fixed descriptor SVM and node-attribute ablations |
| `baselines/` | Tuned descriptor SVM and Graph2Vec-SVM |
| `controls/` | Matched edge controls and application datasets |
| `degree/` | Per-graph degree-tail fitting and bootstrap diagnostics |
| `analysis/` | Performance tables, plots and structural error analysis |

## Installation

Use Python 3.11 and a Linux environment. Create a virtual environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

PyTorch 2.6.0 supports CPU execution and compatible CUDA installations. The neural commands accept `--device cpu` or `--device cuda`. Select one GPU per worker with `CUDA_VISIBLE_DEVICES` when running several workers. The large synthetic graphs require substantially more memory and time than smaller verification datasets.

Degree diagnostics use their own recorded dependency versions; their installation is provided below.

## Synthetic data

The dataset record is <https://zenodo.org/records/16260992>. Download `graph_objects.zip` and `graph_features.zip`, and extract them so that `datasets/raw/` contains `graph_objects/` and `graph_features/`. The converter reads graph pickles; use trusted input archives.

```bash
python data/prepare.py --raw-data datasets/raw --output datasets/synthetic
```

The prepared dataset contains 2,000 graphs, five node descriptors and five graph descriptors. Its manifest records the input files, graph IDs, class labels and file hashes. `data/published_identity.json` identifies the graph arrays used for the reported experiments. Benchmark and control preparation check this identity by default.

To generate a new corpus with the same family parameter ranges:

```bash
python data/generate.py --output datasets/generated_raw --seed 42
python data/prepare.py --raw-data datasets/generated_raw --output datasets/generated
```

Use the explicit `--new-corpus` preparation option for newly generated or otherwise different graph arrays. This fits preprocessing to the new training partitions and records their identity. To reproduce the reported corpus, use inputs that pass the published identity check.

## Main comparison and attribute ablations

Prepare the five stratified 70%|15%|15% partitions and their training-only feature selection and standardisation:

```bash
python benchmark/prepare.py --dataset datasets/synthetic --output runs/benchmark/prepared
```

For a newly generated corpus, use `--dataset datasets/generated --new-corpus` in this command and use that dataset in the subsequent commands.

Run the main GNN comparison and fixed descriptor SVM:

```bash
python benchmark/run.py --stage main --dataset datasets/synthetic --output runs/benchmark --device cuda
```

Run the eight node-attribute conditions with graph descriptors present or zeroed:

```bash
python benchmark/run.py --stage ablation --dataset datasets/synthetic --output runs/benchmark --device cuda
```

Both stages use split seeds **42, 43, 44, 45 and 46**. The main stage performs 30 GNN searches, each with 50 tuning attempts, followed by 30 final GNN fits and five fixed-SVM fits. The ablation stage contains 480 fits. `benchmark/protocol.json` specifies the search space, optimisation and stopping rules. Evaluation uses training-only sequential batch-normalisation calibration. The test partition is evaluated after validation-based model and checkpoint selection.

Commands can be resumed with the same arguments. Completed outputs are verified and unfinished training resumes from its checkpoint. To divide work among processes, assign disjoint `--seeds` or `--architectures` values. Ablations also accept `--conditions` and `--graph-modes` filters. Keep the output directory shared and keep the source unchanged during a run.

## Tuned descriptor and Graph2Vec baselines

These baselines use the benchmark's prepared partitions and selected descriptors:

```bash
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python baselines/run.py prepare --prepared runs/benchmark/prepared --dataset datasets/synthetic --output runs/baselines
for seed in 42 43 44 45 46; do
    python baselines/run.py run --prepared runs/benchmark/prepared --dataset datasets/synthetic --output runs/baselines --seed "$seed"
done
```

Each seed fits a tuned descriptor SVM and Graph2Vec-SVM, completing the nine-condition, 45-fit comparison. Both use the same 25-candidate SVM grid. Graph2Vec uses degree-initialised Weisfeiler–Lehman tokens at levels 0–2, 128-dimensional embeddings, 40 training epochs and 40 inference epochs. Vocabulary and embedding fitting use training graphs. Full settings are in `baselines/protocol.json`.

## Matched controls and application datasets

The control suite includes synthetic full-edge/self-edge comparisons and experiments on NCI1, PROTEINS and IMDB-BINARY. Downloads are verified against the supplied checksums. Preparation constructs the task dependencies and training-only transforms.

```bash
python controls/prepare.py download --downloads datasets/tu
python controls/prepare.py prepare --run runs/controls --synthetic datasets/synthetic/manifest.json --downloads datasets/tu
python controls/work_queue.py worker --run runs/controls --device cuda
```

For a new synthetic corpus, add `--new-corpus --prepared runs/benchmark/prepared` to the preparation command and supply its manifest. This uses the same splits, selected features and scalers as the main benchmark.

Start any number of independent workers against the same run directory. Workers require a shared filesystem with POSIX file locking, and each GPU worker should have its own visible GPU. For example, launch one process with `CUDA_VISIBLE_DEVICES=0` and another with `CUDA_VISIBLE_DEVICES=1`, each using `--device cuda`. Valid tasks are claimed once, completed tasks are verified before dependent tasks run. A worker started after completion exits successfully without starting a training task.

Check progress and collect the results after all workers finish:

```bash
python controls/work_queue.py status --run runs/controls
python controls/work_queue.py is-complete --run runs/controls
python controls/analyse.py --run runs/controls
```

The work queue contains 4,000 logical tasks. Final fits with identical selected settings and initialisation share the benchmark results. Failed tasks retain their execution logs and checkpoints. After resolving an error and stopping the workers, `python controls/work_queue.py retry-failures --run runs/controls` makes failed tasks valid again. Restart workers with the same arguments to continue.

The synthetic controls use the five split seeds above. Application datasets use five grouped outer folds with seed **5102026** and inner split seeds **5102100–5102104**. Each final neural condition has three initialisations; their deterministic seeds are defined in `controls/protocol.py`. The eight configurations combine widths 64/128, dropout 0/0.3 and learning rates 0.001/0.003.

## Degree diagnostics

Use a separate environment for the recorded degree-analysis stack:

```bash
python3.11 -m venv .venv-degree
source .venv-degree/bin/activate
python -m pip install -r degree/requirements.txt
python degree/export_degree_histograms.py --prepared datasets/synthetic --output runs/degree_histograms.zip
python degree/analysis.py --histograms runs/degree_histograms.zip --prepared-manifest datasets/synthetic/manifest.json --output runs/degree --workers 8
python degree/summarise.py --results runs/degree --output runs/degree_summary
```

The full analysis fits all 2,000 graphs with 1,000 bootstrap simulations per graph and compares the discrete power-law tail with exponential and rounded-lognormal alternatives. The seed and fitting settings are specified in `degree/protocol.json`.

## Tables and figures

Reactivate the main environment before collecting the benchmark results:

```bash
source .venv/bin/activate
python analysis/build_publication.py --results runs/benchmark --baselines runs/baselines --output runs/publication
python analysis/analyse_structural_errors.py --results runs/benchmark --baselines runs/baselines --output runs/structural_errors
python analysis/descriptors.py --prepared runs/benchmark/prepared --output runs/descriptors
```

## Verification

The followings support small fixtures test execution, saved predictions, resumption and collection:

```bash
source .venv/bin/activate
python -m unittest discover -s benchmark/tests
python -m unittest discover -s baselines
python -m unittest discover -s analysis
python -m unittest discover -s controls/tests
python benchmark/smoke.py --output runs/benchmark_smoke --device cpu
python controls/tests/smoke_pipeline.py --run runs/controls_smoke
source .venv-degree/bin/activate
python -m unittest discover -s degree
```

Use a new directory for each smoke run. Smoke fixtures use fewer graphs, tasks and epochs. The commands use the full settings in the protocol files.

## Reproducibility

You need to keep the generated manifests, splits, selected features, scalers, task records, checkpoints and prediction files with each run. The code records software versions and verifies input and completion hashes. Furthermore, timings depend on the execution environment. Also, floating-point results can vary across hardware and software builds. Resume training with the same PyTorch version and CPU/CUDA backend. The source package includes configuration and reference identity files required by the workflows. Datasets, trained weights and generated results are created or downloaded by the commands above.
