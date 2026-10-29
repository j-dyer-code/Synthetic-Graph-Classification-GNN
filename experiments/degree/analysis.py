import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import re
import sys
import time
import warnings
import zipfile
import numpy as np
from scipy.special import zeta
from scipy.stats import beta

FAMILIES = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "holme_kim",
]
PROTOCOL_TAG = "degree-distribution-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def immutable_json(path, value):
    if Path(path).exists():
        if canonical(json.loads(Path(path).read_text())) != canonical(value):
            raise ValueError(f"Existing configuration or input differs: {path}")
    else:
        atomic_json(path, value)


def integer(value, label):
    if isinstance(value, bool) or not re.fullmatch("0|[1-9][0-9]*", str(value)):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def validate_graph_histogram(
    meta, degree_values, degree_counts, prepared_graph, class_names
):
    graph_id = str(meta["graph_id"])
    if str(prepared_graph["id"]) != graph_id:
        raise ValueError("Graph ID differs from prepared manifest")
    family = meta["family"]
    label = integer(prepared_graph["label"], "prepared graph label")
    if label >= len(class_names):
        raise ValueError("Prepared graph label is outside the declared family mapping")
    if family != class_names[label]:
        raise ValueError(f"Graph family differs: {graph_id}")
    n = integer(meta["num_nodes"], "num_nodes")
    m = integer(meta["num_edges"], "num_edges")
    if n != prepared_graph["num_nodes"] or m != prepared_graph["undirected_edges"]:
        raise ValueError(f"Graph size differs: {graph_id}")
    values = [integer(x, "degree") for x in degree_values]
    counts = [integer(x, "count") for x in degree_counts]
    if len(values) != len(counts) or not values or len(set(values)) != len(values):
        raise ValueError(
            "Histogram bins must be nonempty, distinct, and paired with counts"
        )
    if any((c == 0 for c in counts)) or n < 2 or any((k >= n for k in values)):
        raise ValueError("Invalid simple-graph histogram support or nonpositive counts")
    if sum(counts) != n or sum((k * c for k, c in zip(values, counts))) != 2 * m:
        raise ValueError(
            f"Histogram count or degree sum does not match graph: {graph_id}"
        )
    if integer(meta["degree_sum"], "degree_sum") != 2 * m:
        raise ValueError("Metadata degree sum does not equal twice the edge count")
    if min(values) != integer(meta["min_degree"], "min_degree") or max(
        values
    ) != integer(meta["max_degree"], "max_degree"):
        raise ValueError("Metadata degree bounds differ")
    if len(values) != integer(meta["distinct_degrees"], "distinct_degrees"):
        raise ValueError("Metadata distinct-degree count differs")
    if not math.isclose(
        float(meta["mean_degree"]), 2 * m / n, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Metadata degree mean differs")
    if not re.fullmatch("[0-9a-f]{64}", meta["degrees_sha256"]):
        raise ValueError("Invalid declared original-node-order degree checksum")
    ordered = sorted(zip(values, counts))
    values, counts = map(list, zip(*ordered))
    identity = dict(
        graph_id=graph_id,
        family=family,
        num_nodes=n,
        num_edges=m,
        degree_values=values,
        degree_counts=counts,
    )
    return dict(
        **identity,
        histogram_sha256=digest(identity),
        original_order_degrees_sha256_declared=meta["degrees_sha256"],
        prepared_graph_sha256=prepared_graph["sha256"],
    )


def load_export(zip_path, prepared_manifest_path, expected_per_family=400):
    zip_path, prepared_manifest_path = (Path(zip_path), Path(prepared_manifest_path))
    prepared = json.loads(prepared_manifest_path.read_text())
    if (
        prepared.get("status") != "complete"
        or prepared.get("schema_version") != 1
        or prepared.get("class_names") != FAMILIES
    ):
        raise ValueError("Unsupported or incomplete prepared manifest")
    refs = {str(g["id"]): g for g in prepared["graphs"]}
    if len(refs) != len(prepared["graphs"]):
        raise ValueError("Duplicate graph IDs in prepared manifest")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        expected = {
            "degree_histograms.csv",
            "graph_metadata.csv",
            "export_manifest.json",
        }
        if len(names) != 3 or set(names) != expected or archive.testzip() is not None:
            raise ValueError(
                "Export ZIP has unexpected/duplicate entries or a CRC failure"
            )
        blobs = {name: archive.read(name) for name in expected}
    manifest = json.loads(blobs["export_manifest.json"])
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete export manifest")
    if (
        manifest["class_names"] != FAMILIES
        or manifest["expected_per_family"] != expected_per_family
    ):
        raise ValueError("Unexpected graph families/count protocol")
    for name in ("degree_histograms.csv", "graph_metadata.csv"):
        if hashlib.sha256(blobs[name]).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Export content checksum mismatch: {name}")
    manifest_hash = file_hash(prepared_manifest_path)
    if manifest["input_manifest_sha256"] != manifest_hash:
        raise ValueError("Export does not reference this prepared manifest")
    graph_hashes = [
        dict(id=str(g["id"]), label=g["label"], sha256=g["sha256"])
        for g in prepared["graphs"]
    ]
    if canonical(graph_hashes) != canonical(
        manifest["source_prepared_graph_checksums"]
    ):
        raise ValueError("Declared prepared-graph checksums differ")
    if (
        digest(dict(manifest_sha256=manifest_hash, graph_files=graph_hashes))
        != manifest["source_dataset_fingerprint"]
    ):
        raise ValueError("Source dataset fingerprint differs")
    if manifest["source_dataset_sha256"] != prepared["dataset_sha256"] or canonical(
        manifest["declared_original_source_files"]
    ) != canonical(prepared["source_files"]):
        raise ValueError("Declared source dataset identity differs")
    metadata_rows = list(
        csv.DictReader(io.StringIO(blobs["graph_metadata.csv"].decode("utf-8")))
    )
    metadata = {row["graph_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or set(metadata) != set(refs):
        raise ValueError("Metadata graph IDs differ or are duplicated")
    hist = defaultdict(list)
    histogram_rows = 0
    for row in csv.DictReader(
        io.StringIO(blobs["degree_histograms.csv"].decode("utf-8"))
    ):
        graph_id = row["graph_id"]
        if graph_id not in metadata or row["family"] != metadata[graph_id]["family"]:
            raise ValueError("Histogram has an unknown graph or inconsistent family")
        hist[graph_id].append(
            (integer(row["degree"], "degree"), integer(row["count"], "count"))
        )
        histogram_rows += 1
    if set(hist) != set(metadata):
        raise ValueError("Some graphs have no histogram")
    records = []
    for graph_id in sorted(metadata, key=lambda s: (len(s), s)):
        values, counts = zip(*hist[graph_id])
        records.append(
            validate_graph_histogram(
                metadata[graph_id], values, counts, refs[graph_id], FAMILIES
            )
        )
    totals = Counter((r["family"] for r in records))
    if (
        totals != {name: expected_per_family for name in FAMILIES}
        or dict(totals) != manifest["counts"]
    ):
        raise ValueError("Graphs per family differ from the declared protocol")
    actual = dict(
        graphs=len(records),
        nodes=sum((r["num_nodes"] for r in records)),
        undirected_edges=sum((r["num_edges"] for r in records)),
        histogram_rows=histogram_rows,
    )
    if any((actual[key] != manifest[key] for key in actual)):
        raise ValueError("Export aggregate counts differ")
    audit = dict(
        **actual,
        counts=dict(totals),
        archive_sha256=file_hash(zip_path),
        prepared_manifest_sha256=manifest_hash,
        export_manifest_sha256=hashlib.sha256(
            blobs["export_manifest.json"]
        ).hexdigest(),
        source_dataset_fingerprint=manifest["source_dataset_fingerprint"],
        histogram_identity_sha256=digest(records),
        checks="ZIP CRC; exported CSV SHA256; graph IDs/families/nodes/edges and declared NPZ hashes against prepared manifest; degree sums and support",
    )
    return (records, audit)


def bootstrap_precision(trials):
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("Bootstrap trials must be a positive integer")
    precision = math.nextafter(0.5 / math.sqrt(trials), 0.0)
    if int(0.25 / precision / precision) != trials:
        raise ValueError(
            "Precision-to-trial conversion did not preserve the requested count"
        )
    return precision


def mc_interval(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials <= 0 or (not 0 < confidence < 1):
        raise ValueError("Invalid binomial interval parameters")
    q = (1 - confidence) / 2
    low = (
        0.0 if successes == 0 else float(beta.ppf(q, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - q, successes + 1, trials - successes))
    )
    return (low, high)


def stable_seed(graph_id):
    return int.from_bytes(
        hashlib.sha256((PROTOCOL_TAG + ":" + str(graph_id)).encode()).digest()[:8],
        "big",
    )


def powerlaw_logpmf(tail, alpha, xmin):
    tail = np.asarray(tail, dtype=np.float64)
    if (
        alpha <= 1
        or xmin < 1
        or (not float(xmin).is_integer())
        or (tail.size == 0)
        or (not np.isfinite(tail).all())
        or (tail < xmin).any()
        or (not np.equal(tail, np.floor(tail)).all())
    ):
        raise ValueError("Invalid discrete power-law support/parameters")
    normalizer = float(zeta(alpha, xmin))
    method = "scipy.special.zeta"
    if not math.isfinite(normalizer) or normalizer <= 0:
        import mpmath

        with mpmath.workdps(80):
            log_normalizer = float(mpmath.log(mpmath.zeta(alpha, xmin)))
        method = "mpmath.zeta 80 decimal digits"
    else:
        log_normalizer = math.log(normalizer)
    result = -alpha * np.log(tail) - log_normalizer
    if not np.isfinite(result).all() or (result > 1e-10).any():
        raise FloatingPointError("Invalid discrete power-law log probabilities")
    return (result, method)


def fit_alternative(tail, log_pl, xmin, name, comparison_threshold=0.1):
    import powerlaw

    cls = {"exponential": powerlaw.Exponential, "lognormal": powerlaw.Lognormal}[name]
    result = dict(distribution=name, status="failed", xmin=int(xmin), n_tail=len(tail))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted = cls(
                data=tail,
                xmin=xmin,
                discrete=True,
                discrete_normalization="round",
                verbose=0,
            )
            raw_parameters = {k: float(v) for k, v in fitted.parameters.items()}
            result.update(
                parameters={
                    k: v if math.isfinite(v) else None
                    for k, v in raw_parameters.items()
                },
                in_range=bool(fitted.in_range()),
                noise_flag=bool(fitted.noise_flag),
            )
            if not all((math.isfinite(v) for v in raw_parameters.values())):
                raise FloatingPointError("Nonfinite alternative parameters")
            probabilities = fitted.pdf(tail)
            floor = 10.0**sys.float_info.min_10_exp
            result["floor_clipped_probability_count"] = int(
                np.count_nonzero(probabilities <= floor)
            )
            if (
                not np.isfinite(probabilities).all()
                or (probabilities <= 0).any()
                or result["floor_clipped_probability_count"]
            ):
                raise FloatingPointError(
                    "Alternative probability underflow, clipping, or invalid mass"
                )
            logs = np.log(probabilities)
            if not np.isfinite(logs).all() or (logs > 1e-10).any():
                raise FloatingPointError(
                    "Nonfinite or positive alternative log probabilities"
                )
            result["log_likelihood"] = float(np.sum(logs))
            if not fitted.in_range() or fitted.noise_flag:
                raise RuntimeError(
                    "Library reports failed convergence or parameters near a boundary"
                )
            parameters = list(result["parameters"].values())
            if (
                not all((math.isfinite(v) for v in parameters))
                or (name == "exponential" and result["parameters"]["Lambda"] <= 0)
                or (name == "lognormal" and result["parameters"]["sigma"] <= 0)
            ):
                raise FloatingPointError("Invalid alternative parameters")
            variance = float(np.var(log_pl - logs))
            result["log_likelihood_difference_variance"] = (
                variance if math.isfinite(variance) else None
            )
            if not math.isfinite(variance) or variance <= 0:
                raise FloatingPointError("Zero or nonfinite likelihood-ratio variance")
            r, p = powerlaw.loglikelihood_ratio(log_pl, logs, nested=False)
            normalized, p_check = powerlaw.loglikelihood_ratio(
                log_pl, logs, nested=False, normalized_ratio=True
            )
            if (
                not all((math.isfinite(float(v)) for v in (r, p, normalized, p_check)))
                or not 0 <= p <= 1
            ):
                raise FloatingPointError("Invalid likelihood-ratio calculation")
            if not math.isclose(float(p), float(p_check), abs_tol=1e-12):
                raise FloatingPointError(
                    "Normalized and raw likelihood-ratio p-values disagree"
                )
            result.update(
                status="complete",
                log_likelihood_ratio=float(r),
                normalized_ratio=float(normalized),
                nominal_p=float(p),
                preference_at_010="inconclusive"
                if p >= comparison_threshold
                else "power_law"
                if r > 0
                else name,
            )
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
    result["warnings"] = [str(w.message) for w in caught]
    return result


def analyse_graph(record, protocol):
    import igraph as ig

    started = time.perf_counter()
    result = dict(
        graph_id=record["graph_id"],
        family=record["family"],
        num_nodes=record["num_nodes"],
        num_edges=record["num_edges"],
        histogram_sha256=record["histogram_sha256"],
        seed=stable_seed(record["graph_id"]),
        status="failed",
    )
    degree = np.repeat(
        np.asarray(record["degree_values"], dtype=np.int64), record["degree_counts"]
    )
    result["degree_summary"] = dict(
        min=int(degree.min()),
        max=int(degree.max()),
        mean=float(degree.mean()),
        population_variance=float(degree.var()),
        distinct=int(len(record["degree_values"])),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if len(np.unique(degree[degree > 0])) < 3:
                raise ValueError(
                    "Fewer than three distinct positive degrees; tail fit is not informative"
                )
            ig.set_random_number_generator(random.Random(result["seed"]))
            fit = ig.power_law_fit(
                degree.tolist(),
                xmin=None,
                method="discrete",
                p_precision=bootstrap_precision(protocol["bootstrap_trials"]),
            )
            if (
                fit.continuous
                or not all(
                    (
                        math.isfinite(x)
                        for x in (fit.alpha, fit.xmin, fit.D, fit.L, fit.p)
                    )
                )
                or fit.alpha <= 1
                or (not 0 <= fit.p <= 1)
            ):
                raise FloatingPointError("Invalid discrete fitted power-law parameters")
            tail = degree[degree >= fit.xmin]
            log_pl, normalizer = powerlaw_logpmf(tail, fit.alpha, fit.xmin)
            ll_difference = float(np.sum(log_pl) - fit.L)
            if not math.isclose(
                float(np.sum(log_pl)), fit.L, rel_tol=1e-09, abs_tol=1e-07
            ):
                raise FloatingPointError(
                    "Independent power-law log likelihood disagrees with igraph"
                )
            b = protocol["bootstrap_trials"]
            k = int(round(fit.p * b))
            if not math.isclose(k / b, fit.p, abs_tol=1e-12):
                raise FloatingPointError(
                    "Bootstrap p does not match the declared replicate count"
                )
            low, high = mc_interval(k, b, protocol["monte_carlo_confidence"])
            result["power_law"] = dict(
                alpha=fit.alpha,
                xmin=int(fit.xmin),
                KS_D=fit.D,
                log_likelihood=fit.L,
                independent_log_likelihood_difference=ll_difference,
                normalizer_method=normalizer,
                n_tail=len(tail),
                n_below_xmin=len(degree) - len(tail),
                tail_fraction=len(tail) / len(degree),
                distinct_tail_degrees=int(len(np.unique(tail))),
                max_degree_over_xmin=float(tail.max() / fit.xmin),
                tail_range_decades=float(math.log10(tail.max() / fit.xmin)),
                insufficient_tail_count=len(tail) < protocol["minimum_tail_count_flag"],
                bootstrap_trials=b,
                bootstrap_exceedances=k,
                nominal_bootstrap_p=fit.p,
                monte_carlo_ci_low=low,
                monte_carlo_ci_high=high,
                monte_carlo_interval_crosses_01=low <= 0.1 <= high,
                nominal_not_rejected_at_01=fit.p >= 0.1,
            )
            result["alternatives"] = {
                name: fit_alternative(
                    tail, log_pl, fit.xmin, name, protocol["comparison_threshold"]
                )
                for name in protocol["alternatives"]
            }
            result["status"] = "complete"
            result["alternative_failures"] = sum(
                (v["status"] != "complete" for v in result["alternatives"].values())
            )
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
    result["warnings"] = [str(w.message) for w in caught]
    result["seconds"] = time.perf_counter() - started
    return result


def checkpoint_task(record, protocol, identity, output):
    result = analyse_graph(record, protocol)
    result["analysis_fingerprint"] = identity
    result["checkpoint_sha256"] = digest(result)
    name = hashlib.sha256(record["graph_id"].encode()).hexdigest() + ".json"
    atomic_json(Path(output) / "graphs" / name, result)
    return (
        result["graph_id"],
        result["status"],
        result.get("alternative_failures", 0),
        result["seconds"],
    )


def verify_checkpoint(checkpoint, record, fingerprint):
    saved = json.loads(Path(checkpoint).read_text())
    checksum = saved.get("checkpoint_sha256")
    content = {k: v for k, v in saved.items() if k != "checkpoint_sha256"}
    if checksum != digest(content):
        raise ValueError("Graph checkpoint content checksum differs")
    if saved.get("analysis_fingerprint") != fingerprint:
        raise ValueError("Graph checkpoint has different data/code/protocol")
    for key in ("graph_id", "family", "num_nodes", "num_edges", "histogram_sha256"):
        if saved.get(key) != record[key]:
            raise ValueError(f"Graph checkpoint input identity differs: {key}")
    if saved.get("status") == "complete":
        if saved.get("power_law", {}).get("bootstrap_trials") != 1000 or set(
            saved.get("alternatives", {})
        ) != {"exponential", "lognormal"}:
            raise ValueError("Graph checkpoint completion payload is incomplete")
    return saved


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if (
        protocol.get("protocol") != PROTOCOL_TAG
        or protocol.get("bootstrap_trials") != 1000
        or protocol.get("alternatives") != ["exponential", "lognormal"]
    ):
        raise ValueError(
            "The protocol requires 1,000 trials and the declared alternatives"
        )
    if (
        protocol.get("minimum_tail_count_flag") != 50
        or protocol.get("monte_carlo_confidence") != 0.95
    ):
        raise ValueError("Unexpected support/Monte Carlo settings")
    if (
        protocol.get("goodness_of_fit_threshold") != 0.1
        or protocol.get("comparison_threshold") != 0.1
    ):
        raise ValueError("The declared nominal descriptive thresholds are 0.1")
    return protocol


def main():
    parser = argparse.ArgumentParser(
        description="Fit discrete degree-distribution tails and bootstrap diagnostics for each graph."
    )
    parser.add_argument("--histograms", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol.json")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocol = load_protocol(args.protocol)
    records, audit = load_export(args.histograms, args.prepared_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    software = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "igraph", "powerlaw", "mpmath")
    }
    import igraph as ig

    if (
        software["igraph"] != "0.11.9"
        or ig.__igraph_version__ != "0.10.16"
        or software["powerlaw"] != "2.0.0"
    ):
        raise ValueError("Use igraph 0.11.9 with C core 0.10.16 and powerlaw 2.0.0")
    identity = dict(
        protocol_sha256=digest(protocol),
        input_audit_sha256=digest(audit),
        analysis_code_sha256=file_hash(__file__),
        software=software,
        igraph_core=ig.__igraph_version__,
    )
    fingerprint = digest(identity)
    print(
        f"Validated {len(records)} graphs; {audit['nodes']} nodes; fingerprint={fingerprint}",
        flush=True,
    )
    lock = args.output / ".analysis.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
    except FileExistsError as error:
        raise RuntimeError(
            "Analysis is locked; verify the previous process stopped before removing its lock"
        ) from error
    with os.fdopen(fd, "w") as handle:
        json.dump(dict(pid=os.getpid(), host=platform.node()), handle)
    try:
        immutable_json(args.output / "protocol.json", protocol)
        immutable_json(args.output / "input_audit.json", audit)
        immutable_json(args.output / "analysis_identity.json", identity)
        if args.validate_only:
            return
        previous_report = args.output / "completion_report.json"
        if previous_report.exists():
            previous = json.loads(previous_report.read_text())
            if previous["analysis_fingerprint"] != fingerprint:
                raise ValueError(
                    "Existing completion report has a different analysis fingerprint"
                )
            for name, checksum in previous["result_files"].items():
                if (
                    Path(name).name != name
                    or file_hash(args.output / "graphs" / name) != checksum
                ):
                    raise ValueError(
                        "Completed graph output differs from completion-report checksum"
                    )
        pending = []
        for record in records:
            name = hashlib.sha256(record["graph_id"].encode()).hexdigest() + ".json"
            checkpoint = args.output / "graphs" / name
            if checkpoint.exists():
                old = verify_checkpoint(checkpoint, record, fingerprint)
                if old["status"] == "complete":
                    continue
            pending.append(record)
        attempt = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version,
            platform=platform.platform(),
            workers=args.workers,
            threads_per_process=1,
            command=sys.argv,
            pending_graphs=len(pending),
            analysis_fingerprint=fingerprint,
        )
        atomic_json(
            args.output
            / (
                "attempt_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                + ".json"
            ),
            attempt,
        )
        started = time.perf_counter()
        print(
            f"Starting {len(pending)} pending graphs with {args.workers} processes",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(checkpoint_task, r, protocol, fingerprint, str(args.output))
                for r in pending
            ]
            for done, future in enumerate(as_completed(futures), 1):
                graph_id, status, alt_failures, seconds = future.result()
                if done % 20 == 0 or status != "complete" or done == len(pending):
                    print(
                        f"completed={done}/{len(pending)} graph={graph_id} status={status} alternatives_flagged={alt_failures} graph_seconds={seconds:.2f} elapsed_seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
        results = []
        for record in records:
            name = hashlib.sha256(record["graph_id"].encode()).hexdigest() + ".json"
            results.append(
                verify_checkpoint(args.output / "graphs" / name, record, fingerprint)
            )
        report = dict(
            status="complete"
            if all((r["status"] == "complete" for r in results))
            else "complete_with_fit_failures",
            expected_graphs=len(records),
            recorded_graphs=len(results),
            powerlaw_fit_failures=sum((r["status"] != "complete" for r in results)),
            alternative_fit_failures=sum(
                (r.get("alternative_failures", 0) for r in results)
            ),
            analysis_fingerprint=fingerprint,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            result_files={
                p.name: file_hash(p)
                for p in sorted((args.output / "graphs").glob("*.json"))
            },
        )
        atomic_json(args.output / "completion_report.json", report)
        print(
            json.dumps({k: v for k, v in report.items() if k != "result_files"}),
            flush=True,
        )
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
