import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import numpy as np

FAMILIES = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "holme_kim",
]
ALTERNATIVES = ["exponential", "lognormal"]
PER_FAMILY = 400
HEX = re.compile("^[0-9a-f]{64}$")
PL_FIELDS = [
    "alpha",
    "xmin",
    "KS_D",
    "log_likelihood",
    "independent_log_likelihood_difference",
    "normalizer_method",
    "n_tail",
    "n_below_xmin",
    "tail_fraction",
    "distinct_tail_degrees",
    "max_degree_over_xmin",
    "tail_range_decades",
    "insufficient_tail_count",
    "bootstrap_trials",
    "bootstrap_exceedances",
    "nominal_bootstrap_p",
    "monte_carlo_ci_low",
    "monte_carlo_ci_high",
    "monte_carlo_interval_crosses_01",
    "nominal_not_rejected_at_01",
]
DEGREE_FIELDS = ["min", "max", "mean", "population_variance", "distinct"]
GRAPH_FIELDS = (
    [
        "graph_id",
        "family",
        "num_nodes",
        "num_edges",
        "fit_status",
        "seed",
        "seconds",
        "histogram_sha256",
        "checkpoint_sha256",
        "error_type",
        "error",
        "warnings_json",
    ]
    + ["degree_" + k for k in DEGREE_FIELDS]
    + ["pl_" + k for k in PL_FIELDS]
)
ALT_FIELDS = [
    "graph_id",
    "family",
    "power_law_status",
    "distribution",
    "status",
    "xmin",
    "n_tail",
    "pl_nominal_bootstrap_p",
    "pl_insufficient_tail_count",
    "pl_adequate_and_nominal_not_rejected",
    "parameters_json",
    "in_range",
    "noise_flag",
    "floor_clipped_probability_count",
    "log_likelihood",
    "log_likelihood_difference_variance",
    "log_likelihood_ratio",
    "normalized_ratio",
    "nominal_p",
    "preference_at_010",
    "error_type",
    "error",
    "warnings_json",
]
QUANTITIES = [
    "alpha",
    "xmin",
    "n_tail",
    "tail_fraction",
    "max_degree_over_xmin",
    "tail_range_decades",
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name, low=None, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its valid range")
    return value


def close(actual, expected, name, abs_tol=1e-10):
    if not math.isclose(actual, expected, rel_tol=1e-09, abs_tol=abs_tol):
        raise ValueError(f"{name} disagrees with its defining calculation")


def validate_result(result, fingerprint):
    if result.get("analysis_fingerprint") != fingerprint:
        raise ValueError("Checkpoint has a different analysis fingerprint")
    checksum = result.get("checkpoint_sha256")
    if checksum != digest(
        {k: v for k, v in result.items() if k != "checkpoint_sha256"}
    ):
        raise ValueError("Embedded checkpoint checksum differs")
    gid = result.get("graph_id")
    if not isinstance(gid, str) or not re.fullmatch("0|[1-9][0-9]*", gid):
        raise ValueError("Graph ID must be a canonical nonnegative integer string")
    if result.get("family") not in FAMILIES or result.get("status") not in {
        "complete",
        "failed",
    }:
        raise ValueError("Unknown graph family or fit status")
    n = integer(result["num_nodes"], "num_nodes", 2)
    m = integer(result["num_edges"], "num_edges")
    if m > n * (n - 1) // 2:
        raise ValueError("Edge count exceeds a simple graph's bound")
    if not HEX.fullmatch(result.get("histogram_sha256", "")):
        raise ValueError("Invalid histogram checksum")
    integer(result["seed"], "seed")
    finite(result["seconds"], "seconds", 0)
    ds = result["degree_summary"]
    for key in ("min", "max", "distinct"):
        integer(ds[key], "degree_" + key)
    if not 0 <= ds["min"] <= ds["max"] < n or not 1 <= ds["distinct"] <= n:
        raise ValueError("Invalid degree summary bounds")
    close(finite(ds["mean"], "degree_mean", 0), 2 * m / n, "degree mean")
    finite(ds["population_variance"], "degree variance", 0)
    if result["status"] == "failed":
        if not result.get("error") or not result.get("error_type"):
            raise ValueError("A failed graph must retain its failure reason")
        return
    pl = result["power_law"]
    if any((k not in pl for k in PL_FIELDS)):
        raise ValueError("Completed power-law payload is missing fields")
    if finite(pl["alpha"], "alpha") <= 1:
        raise ValueError("Power-law alpha must exceed one")
    xmin = integer(pl["xmin"], "xmin", 1)
    ntail = integer(pl["n_tail"], "n_tail", 1)
    if (
        xmin > ds["max"]
        or ntail > n
        or integer(pl["n_below_xmin"], "n_below_xmin") != n - ntail
    ):
        raise ValueError("Invalid fitted tail bounds/counts")
    if (
        not 1
        <= integer(pl["distinct_tail_degrees"], "distinct_tail_degrees", 1)
        <= ntail
    ):
        raise ValueError("Invalid distinct tail count")
    close(
        finite(pl["tail_fraction"], "tail_fraction", 0, 1), ntail / n, "tail fraction"
    )
    close(
        finite(pl["max_degree_over_xmin"], "tail range", 1),
        ds["max"] / xmin,
        "tail range",
    )
    close(
        finite(pl["tail_range_decades"], "tail decades", 0),
        math.log10(ds["max"] / xmin),
        "tail decades",
    )
    finite(pl["KS_D"], "KS discrepancy", 0, 1)
    finite(pl["log_likelihood"], "power-law likelihood", high=1e-08)
    finite(
        pl["independent_log_likelihood_difference"], "likelihood crosscheck difference"
    )
    if pl["bootstrap_trials"] != 1000:
        raise ValueError("The study requires exactly 1,000 bootstrap replicates")
    k = integer(pl["bootstrap_exceedances"], "bootstrap exceedances")
    if k > 1000:
        raise ValueError("Too many bootstrap exceedances")
    p = finite(pl["nominal_bootstrap_p"], "bootstrap p", 0, 1)
    close(p, k / 1000, "bootstrap p", abs_tol=1e-12)
    lo = finite(pl["monte_carlo_ci_low"], "MC lower bound", 0, 1)
    hi = finite(pl["monte_carlo_ci_high"], "MC upper bound", 0, 1)
    if not lo <= p <= hi:
        raise ValueError("Invalid Monte Carlo interval ordering")
    flags = {
        "insufficient_tail_count": ntail < 50,
        "nominal_not_rejected_at_01": p >= 0.1,
        "monte_carlo_interval_crosses_01": lo <= 0.1 <= hi,
    }
    if any(
        (
            not isinstance(pl[key], bool) or pl[key] != expected
            for key, expected in flags.items()
        )
    ):
        raise ValueError("Power-law interpretation flags disagree with recorded values")
    alternatives = result.get("alternatives", {})
    if set(alternatives) != set(ALTERNATIVES):
        raise ValueError("Completed graph must retain both alternative outcomes")
    alt_failures = 0
    for name, alternative in alternatives.items():
        if alternative.get("distribution") != name or alternative.get("status") not in {
            "complete",
            "failed",
        }:
            raise ValueError("Invalid alternative distribution/status")
        if alternative["xmin"] != xmin or alternative["n_tail"] != ntail:
            raise ValueError("Alternative comparison does not use the same tail")
        if alternative["status"] == "failed":
            alt_failures += 1
            if not alternative.get("error_type") or not alternative.get("error"):
                raise ValueError("Failed alternative must retain its failure reason")
            continue
        if (
            alternative.get("noise_flag") is not False
            or alternative.get("in_range") is not True
            or alternative.get("floor_clipped_probability_count") != 0
        ):
            raise ValueError("Completed alternative retains an invalid-fit flag")
        params = alternative["parameters"]
        for key, value in params.items():
            finite(value, "alternative parameter " + key)
        if (
            name == "exponential"
            and params.get("Lambda", 0) <= 0
            or (name == "lognormal" and params.get("sigma", 0) <= 0)
        ):
            raise ValueError("Alternative rate/scale must be positive")
        ll = finite(alternative["log_likelihood"], "alternative likelihood", high=1e-08)
        r = finite(alternative["log_likelihood_ratio"], "likelihood ratio")
        close(
            r,
            pl["log_likelihood"] + pl["independent_log_likelihood_difference"] - ll,
            "likelihood ratio",
            abs_tol=1e-07,
        )
        variance = finite(
            alternative["log_likelihood_difference_variance"], "LR variance", 0
        )
        if variance == 0:
            raise ValueError("A completed likelihood ratio cannot have zero variance")
        norm = finite(alternative["normalized_ratio"], "normalized likelihood ratio")
        close(norm, r / math.sqrt(ntail * variance), "normalized likelihood ratio")
        p = finite(alternative["nominal_p"], "LR p", 0, 1)
        close(p, math.erfc(abs(norm) / math.sqrt(2)), "LR p")
        expected = "inconclusive" if p >= 0.1 else "power_law" if r > 0 else name
        if alternative.get("preference_at_010") != expected:
            raise ValueError(
                "Alternative preference disagrees with the nominal 0.10 threshold"
            )
    if result.get("alternative_failures") != alt_failures:
        raise ValueError("Per-graph alternative failure count disagrees")


def load_completed_results(root):
    root = Path(root)
    if (root / ".analysis.lock").exists():
        raise ValueError("Analysis still has a lock; wait for it to finish")
    report = read_json(root / "completion_report.json")
    audit = read_json(root / "input_audit.json")
    identity = read_json(root / "analysis_identity.json")
    protocol = read_json(root / "protocol.json")
    if report.get("status") not in {"complete", "complete_with_fit_failures"}:
        raise ValueError(
            "A completed study is required; partial summaries are disabled"
        )
    expected_graphs = PER_FAMILY * len(FAMILIES)
    counts = {family: PER_FAMILY for family in FAMILIES}
    if (
        report.get("expected_graphs") != expected_graphs
        or report.get("recorded_graphs") != expected_graphs
        or audit.get("graphs") != expected_graphs
        or (audit.get("counts") != counts)
    ):
        raise ValueError("Expected exactly 2,000 graphs and 400 per family")
    if identity.get("protocol_sha256") != digest(protocol) or identity.get(
        "input_audit_sha256"
    ) != digest(audit):
        raise ValueError(
            "Protocol or input-audit checksum differs from analysis identity"
        )
    if (
        protocol.get("protocol") != "degree-distribution-v1"
        or protocol.get("bootstrap_trials") != 1000
        or protocol.get("minimum_tail_count_flag") != 50
        or (protocol.get("goodness_of_fit_threshold") != 0.1)
        or (protocol.get("comparison_threshold") != 0.1)
        or (protocol.get("monte_carlo_confidence") != 0.95)
        or (protocol.get("alternatives") != ALTERNATIVES)
    ):
        raise ValueError("Unexpected statistical protocol")
    fingerprint = digest(identity)
    if report.get("analysis_fingerprint") != fingerprint:
        raise ValueError("Completion report fingerprint differs from analysis identity")
    files = report.get("result_files")
    if not isinstance(files, dict) or len(files) != expected_graphs:
        raise ValueError("Completion report must list all 2,000 checkpoint files")
    actual_names = {p.name for p in (root / "graphs").glob("*.json")}
    if actual_names != set(files):
        raise ValueError("Checkpoint directory and completion file list differ")
    results, ids = ([], set())
    for name, checksum in sorted(files.items()):
        if (
            not re.fullmatch("[0-9a-f]{64}\\.json", name)
            or not isinstance(checksum, str)
            or (not HEX.fullmatch(checksum))
        ):
            raise ValueError("Invalid checkpoint filename/checksum declaration")
        path = root / "graphs" / name
        if path.is_symlink() or file_hash(path) != checksum:
            raise ValueError("Checkpoint file checksum differs from completion report")
        result = read_json(path)
        validate_result(result, fingerprint)
        gid = result["graph_id"]
        if gid in ids or name != hashlib.sha256(gid.encode()).hexdigest() + ".json":
            raise ValueError("Duplicate graph ID or incorrect checkpoint filename")
        ids.add(gid)
        results.append(result)
    results.sort(key=lambda r: int(r["graph_id"]))
    if dict(Counter((r["family"] for r in results))) != counts:
        raise ValueError("Recorded results do not contain 400 graphs per family")
    if (
        sum((r["num_nodes"] for r in results)) != audit["nodes"]
        or sum((r["num_edges"] for r in results)) != audit["undirected_edges"]
    ):
        raise ValueError("Result graph sizes disagree with input-audit totals")
    failed = sum((r["status"] == "failed" for r in results))
    alt_failed = sum((r.get("alternative_failures", 0) for r in results))
    if (
        report.get("powerlaw_fit_failures") != failed
        or report.get("alternative_fit_failures") != alt_failed
    ):
        raise ValueError("Completion-report failure counts disagree")
    expected_status = "complete_with_fit_failures" if failed else "complete"
    if report["status"] != expected_status:
        raise ValueError("Completion status disagrees with recorded power-law failures")
    return (
        results,
        dict(report=report, input_audit=audit, identity=identity, protocol=protocol),
    )


def table_rows(results):
    graph_rows, alternative_rows = ([], [])
    for result in results:
        row = {
            key: result.get(key)
            for key in GRAPH_FIELDS
            if not key.startswith(("degree_", "pl_"))
        }
        row.update(
            fit_status=result["status"],
            warnings_json=canonical(result.get("warnings", [])),
        )
        row.update(
            {
                "degree_" + key: result["degree_summary"].get(key)
                for key in DEGREE_FIELDS
            }
        )
        valid = result["status"] == "complete"
        pl = result.get("power_law", {}) if valid else {}
        row.update({"pl_" + key: pl.get(key) for key in PL_FIELDS})
        graph_rows.append(row)
        for name in ALTERNATIVES:
            alternative = result.get("alternatives", {}).get(name, {}) if valid else {}
            item = {key: alternative.get(key) for key in ALT_FIELDS}
            item.update(
                graph_id=result["graph_id"],
                family=result["family"],
                power_law_status=result["status"],
                distribution=name,
                status=alternative.get(
                    "status", "unavailable_due_to_power_law_failure"
                ),
                pl_nominal_bootstrap_p=pl.get("nominal_bootstrap_p"),
                pl_insufficient_tail_count=pl.get("insufficient_tail_count"),
                pl_adequate_and_nominal_not_rejected=not pl["insufficient_tail_count"]
                and pl["nominal_not_rejected_at_01"]
                if valid
                else None,
                parameters_json=canonical(alternative.get("parameters", {})),
                warnings_json=canonical(alternative.get("warnings", [])),
            )
            if not valid:
                item.update(
                    error_type=result.get("error_type"), error=result.get("error")
                )
            alternative_rows.append(item)
    family_rows = []
    for family in FAMILIES:
        members = [r for r in results if r["family"] == family]
        valid = [r for r in members if r["status"] == "complete"]
        adequate = [r for r in valid if not r["power_law"]["insufficient_tail_count"]]
        eligible = [r for r in adequate if r["power_law"]["nominal_not_rejected_at_01"]]
        row = dict(
            family=family,
            graphs_expected=PER_FAMILY,
            graphs_recorded=len(members),
            pl_valid_count=len(valid),
            pl_failed_count=len(members) - len(valid),
            pl_insufficient_tail_count=len(valid) - len(adequate),
            pl_adequate_tail_count=len(adequate),
            pl_nominal_not_rejected_count_all_valid=sum(
                (r["power_law"]["nominal_not_rejected_at_01"] for r in valid)
            ),
            pl_nominal_not_rejected_count_adequate=len(eligible),
            pl_mc_interval_crosses_01_count_all_valid=sum(
                (r["power_law"]["monte_carlo_interval_crosses_01"] for r in valid)
            ),
            pl_mc_interval_crosses_01_count_adequate=sum(
                (r["power_law"]["monte_carlo_interval_crosses_01"] for r in adequate)
            ),
            pl_quantile_denominator=len(valid),
        )
        for quantity in QUANTITIES:
            values = [r["power_law"][quantity] for r in valid]
            quantiles = (
                np.quantile(values, [0.25, 0.5, 0.75], method="linear").tolist()
                if values
                else [None, None, None]
            )
            row.update(
                {
                    f"pl_{quantity}_{suffix}_all_valid": value
                    for suffix, value in zip(("q25", "median", "q75"), quantiles)
                }
            )
        for name in ALTERNATIVES:
            for suffix, subset in [
                ("all_valid_pl", valid),
                ("adequate_and_not_rejected_pl", eligible),
            ]:
                comparisons = [r["alternatives"][name] for r in subset]
                good = [a for a in comparisons if a["status"] == "complete"]
                prefix = name + "_" + suffix + "_"
                row[prefix + "denominator"] = len(subset)
                row[prefix + "valid_count"] = len(good)
                row[prefix + "failed_count"] = len(comparisons) - len(good)
                for label, preference in [
                    ("prefer_pl", "power_law"),
                    ("prefer_alternative", name),
                    ("inconclusive", "inconclusive"),
                ]:
                    row[prefix + label + "_count"] = sum(
                        (a["preference_at_010"] == preference for a in good)
                    )
            row[name + "_unavailable_due_to_pl_failure_count"] = len(members) - len(
                valid
            )
        family_rows.append(row)
    return (graph_rows, alternative_rows, family_rows)


def column_definitions(family_columns):
    graph = {key: "Checkpoint field: " + key for key in GRAPH_FIELDS}
    graph.update(
        fit_status="complete or failed; failed rows remain present",
        warnings_json="JSON list of recorded warnings",
        error="Recorded failure reason, if any",
        error_type="Recorded failure class, if any",
    )
    for key in DEGREE_FIELDS:
        graph["degree_" + key] = "Empirical within-graph degree summary: " + key
    for key in PL_FIELDS:
        graph["pl_" + key] = (
            "Power-law checkpoint field: " + key + "; blank if power-law fit failed"
        )
    graph["pl_nominal_bootstrap_p"] = "Strict bootstrap exceedances / 1000"
    graph["pl_insufficient_tail_count"] = "True when n_tail < 50"
    graph[
        "pl_monte_carlo_ci_low"
    ] = "Lower 95% Clopper-Pearson bound for bootstrap exceedance probability, conditional on the diagnostic"
    graph[
        "pl_monte_carlo_ci_high"
    ] = "Upper 95% Clopper-Pearson bound for bootstrap exceedance probability, conditional on the diagnostic"
    alt = {key: "Alternative comparison checkpoint field: " + key for key in ALT_FIELDS}
    alt.update(
        status="complete, failed, or unavailable_due_to_power_law_failure",
        parameters_json="JSON object of fitted parameters, including retained nulls for numerical failures",
        preference_at_010="inconclusive when nominal_p >= 0.10; otherwise power_law for positive R or alternative name for nonpositive R",
        log_likelihood_ratio="Sum log p_power_law minus sum log p_alternative on the identical fitted tail",
        pl_adequate_and_nominal_not_rejected="True only when power-law n_tail >= 50 and its nominal bootstrap p >= 0.10",
        nominal_p="Two-sided nominal non-nested likelihood-ratio p; distinct from bootstrap p",
    )
    family = {}
    for key in family_columns:
        if key == "family":
            meaning = "Original graph generator family"
        elif "_all_valid" in key and any(
            (s in key for s in ("_q25_", "_median_", "_q75_"))
        ):
            meaning = "Per-graph fitted quantity over all completed power-law fits; linear-interpolation quantile; denominator pl_quantile_denominator"
        elif "adequate_and_not_rejected_pl" in key:
            meaning = "Alternative comparison count within graphs having n_tail >= 50 and nominal bootstrap p >= 0.10; denominator explicitly recorded in the matching prefix"
        elif "all_valid_pl" in key:
            meaning = "Alternative comparison count within all completed power-law fits; denominator explicitly recorded in the matching prefix"
        elif "unavailable" in key:
            meaning = "No alternative comparison available because the graph's power-law fit failed"
        else:
            meaning = {
                "graphs_expected": "Required graphs per family (400)",
                "graphs_recorded": "All graphs retained, including failures",
                "pl_valid_count": "Graphs with completed power-law fits",
                "pl_failed_count": "Graphs with failed power-law fits",
                "pl_insufficient_tail_count": "Completed power-law fits with n_tail < 50",
                "pl_adequate_tail_count": "Completed power-law fits with n_tail >= 50",
                "pl_nominal_not_rejected_count_all_valid": "Completed power-law fits with nominal bootstrap p >= 0.10; denominator pl_valid_count",
                "pl_nominal_not_rejected_count_adequate": "Completed power-law fits with n_tail >= 50 and nominal bootstrap p >= 0.10; denominator pl_adequate_tail_count",
                "pl_mc_interval_crosses_01_count_all_valid": "Completed fits whose 95% Monte Carlo interval includes 0.10; denominator pl_valid_count",
                "pl_mc_interval_crosses_01_count_adequate": "Adequate-tail fits whose 95% Monte Carlo interval includes 0.10; denominator pl_adequate_tail_count",
                "pl_quantile_denominator": "Number of completed power-law fits entering every reported family quantile",
            }[key]
        family[key] = meaning
    return {
        "per_graph_fits.csv": graph,
        "alternative_comparisons.csv": alt,
        "family_summary.csv": family,
    }


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarise(results, output):
    results, output = (Path(results), Path(output))
    if output.exists():
        raise FileExistsError("Summary output already exists; choose a new directory")
    records, source = load_completed_results(results)
    graph_rows, alternative_rows, family_rows = table_rows(records)
    family_columns = list(family_rows[0])
    definitions = column_definitions(family_columns)
    output.mkdir(parents=True, exist_ok=False)
    try:
        tables = [
            ("per_graph_fits.csv", graph_rows, GRAPH_FIELDS),
            ("alternative_comparisons.csv", alternative_rows, ALT_FIELDS),
            ("family_summary.csv", family_rows, family_columns),
        ]
        for filename, rows, columns in tables:
            write_csv(output / filename, rows, columns)
        summary_audit = dict(
            status=source["report"]["status"],
            purpose="Per-graph degree-distribution summaries",
            created_utc=datetime.now(timezone.utc).isoformat(),
            analysis_fingerprint=source["report"]["analysis_fingerprint"],
            expected_graphs=2000,
            recorded_graphs=len(records),
            graphs_per_family=PER_FAMILY,
            powerlaw_fit_failures=source["report"]["powerlaw_fit_failures"],
            alternative_fit_failures=source["report"]["alternative_fit_failures"],
            alternative_comparisons_unavailable_due_to_powerlaw_failure=2
            * source["report"]["powerlaw_fit_failures"],
            input_files={
                name: file_hash(results / name)
                for name in (
                    "completion_report.json",
                    "analysis_identity.json",
                    "input_audit.json",
                    "protocol.json",
                )
            },
            summarizer_sha256=file_hash(__file__),
            validation="Completion-report SHA256 for every result; embedded checkpoint digest; one verified analysis fingerprint; protocol/audit identity; unique graph IDs; exactly 2000 graphs and 400 per family; input size totals; result arithmetic and fit flags; completeness/failure totals",
            csv_conventions=dict(
                blank="Quantity unavailable, usually because the fit failed",
                booleans="True / False",
                quantiles="numpy.quantile method='linear', separately within each family, over completed power-law fits",
                comparison_threshold=0.1,
                goodness_of_fit_threshold=0.1,
                adequate_tail_minimum_count=50,
            ),
            columns=definitions,
            output_files={
                name: dict(rows=len(rows), sha256=file_hash(output / name))
                for name, rows, _ in tables
            },
        )
        (output / "summary_audit.json").write_text(
            json.dumps(summary_audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return summary_audit


def main():
    parser = argparse.ArgumentParser(
        description="Validate completed degree diagnostics and export per-graph and family summary tables."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = summarise(args.results, args.output)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "status",
                    "recorded_graphs",
                    "powerlaw_fit_failures",
                    "alternative_fit_failures",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
