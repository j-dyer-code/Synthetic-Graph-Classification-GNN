from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import summarise as summary


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def artificial_result(gid, family, fingerprint):
    small = gid == 1
    ntail = 25 if small else 100
    edges = 100 if small else 200
    result = dict(
        graph_id=str(gid),
        family=family,
        num_nodes=100,
        num_edges=edges,
        seed=gid,
        seconds=0.01,
        histogram_sha256="a" * 64,
        analysis_fingerprint=fingerprint,
        status="failed" if gid == 0 else "complete",
        warnings=[],
        degree_summary=dict(
            min=1 if small else 2,
            max=10,
            mean=2 * edges / 100,
            population_variance=1.0,
            distinct=5,
        ),
    )
    if gid == 0:
        result.update(
            error_type="ArtificialFailure", error="Intentional software fixture failure"
        )
    else:
        result["power_law"] = dict(
            alpha=4.5 if small else 2.5,
            xmin=2,
            KS_D=0.05,
            log_likelihood=-100.0,
            independent_log_likelihood_difference=0.0,
            normalizer_method="ARTIFICIAL FIXTURE",
            n_tail=ntail,
            n_below_xmin=100 - ntail,
            tail_fraction=ntail / 100,
            distinct_tail_degrees=4,
            max_degree_over_xmin=5.0,
            tail_range_decades=math.log10(5),
            insufficient_tail_count=small,
            bootstrap_trials=1000,
            bootstrap_exceedances=100,
            nominal_bootstrap_p=0.1,
            monte_carlo_ci_low=0.08,
            monte_carlo_ci_high=0.12,
            monte_carlo_interval_crosses_01=True,
            nominal_not_rejected_at_01=True,
        )
        alternatives = {}
        for name in summary.ALTERNATIVES:
            r = (
                2.0
                if name == "exponential" and gid == 3
                else -2.0
                if name == "lognormal" and gid == 4
                else 1.0
            )
            variance = 1.0 / ntail
            normalized = r
            p = math.erfc(abs(normalized) / math.sqrt(2))
            alternative = dict(
                distribution=name,
                status="complete",
                xmin=2,
                n_tail=ntail,
                parameters={"Lambda": 0.5}
                if name == "exponential"
                else {"mu": 1.0, "sigma": 0.5},
                in_range=True,
                noise_flag=False,
                floor_clipped_probability_count=0,
                log_likelihood=-100.0 - r,
                log_likelihood_difference_variance=variance,
                log_likelihood_ratio=r,
                normalized_ratio=normalized,
                nominal_p=p,
                preference_at_010="inconclusive"
                if p >= 0.1
                else "power_law"
                if r > 0
                else name,
                warnings=[],
            )
            if gid == 2 and name == "lognormal":
                alternative = dict(
                    distribution=name,
                    status="failed",
                    xmin=2,
                    n_tail=ntail,
                    parameters={"mu": None, "sigma": None},
                    noise_flag=True,
                    error_type="ArtificialFailure",
                    error="Intentional failed alternative",
                    warnings=[],
                )
            alternatives[name] = alternative
        result["alternatives"] = alternatives
        result["alternative_failures"] = int(gid == 2)
    result["checkpoint_sha256"] = summary.digest(result)
    return result


def cohort_fixture(root):
    root.mkdir()
    (root / "graphs").mkdir()
    protocol = dict(
        protocol="degree-distribution-v1",
        bootstrap_trials=1000,
        minimum_tail_count_flag=50,
        goodness_of_fit_threshold=0.1,
        comparison_threshold=0.1,
        monte_carlo_confidence=0.95,
        alternatives=summary.ALTERNATIVES,
    )
    audit = dict(
        graphs=2000,
        counts={f: 400 for f in summary.FAMILIES},
        nodes=200000,
        undirected_edges=399900,
        purpose="software_test",
    )
    identity = dict(
        protocol_sha256=summary.digest(protocol),
        input_audit_sha256=summary.digest(audit),
        analysis_code_sha256="b" * 64,
        purpose="software_test",
    )
    fingerprint = summary.digest(identity)
    for gid in range(2000):
        result = artificial_result(gid, summary.FAMILIES[gid // 400], fingerprint)
        name = hashlib.sha256(str(gid).encode()).hexdigest() + ".json"
        write_json(root / "graphs" / name, result)
    report = dict(
        status="complete_with_fit_failures",
        expected_graphs=2000,
        recorded_graphs=2000,
        powerlaw_fit_failures=1,
        alternative_fit_failures=1,
        analysis_fingerprint=fingerprint,
        result_files={
            p.name: summary.file_hash(p)
            for p in sorted((root / "graphs").glob("*.json"))
        },
    )
    for name, value in [
        ("protocol.json", protocol),
        ("input_audit.json", audit),
        ("analysis_identity.json", identity),
        ("completion_report.json", report),
    ]:
        write_json(root / name, value)


class SummaryIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.results = cls.root / "results"
        cohort_fixture(cls.results)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @contextmanager
    def changed_checkpoint(self, gid, mutate, embedded=False, outer=False):
        path = (
            self.results
            / "graphs"
            / (hashlib.sha256(str(gid).encode()).hexdigest() + ".json")
        )
        report_path = self.results / "completion_report.json"
        old, old_report = (path.read_bytes(), report_path.read_bytes())
        try:
            value = json.loads(old)
            mutate(value)
            if embedded:
                value["checkpoint_sha256"] = summary.digest(
                    {k: v for k, v in value.items() if k != "checkpoint_sha256"}
                )
            write_json(path, value)
            if outer:
                report = json.loads(old_report)
                report["result_files"][path.name] = summary.file_hash(path)
                write_json(report_path, report)
            yield
        finally:
            path.write_bytes(old)
            report_path.write_bytes(old_report)

    def test_complete_cohort_retains_failed_graph_and_alternative(self):
        output = self.root / "summary"
        audit = summary.summarise(self.results, output)
        self.assertEqual(audit["recorded_graphs"], 2000)
        self.assertEqual(audit["powerlaw_fit_failures"], 1)
        self.assertEqual(audit["alternative_fit_failures"], 1)
        self.assertEqual(
            audit["alternative_comparisons_unavailable_due_to_powerlaw_failure"], 2
        )
        with (output / "per_graph_fits.csv").open() as handle:
            graphs = list(csv.DictReader(handle))
        self.assertEqual(len(graphs), 2000)
        self.assertEqual(graphs[0]["fit_status"], "failed")
        self.assertEqual(graphs[0]["pl_alpha"], "")
        self.assertIn("Intentional", graphs[0]["error"])
        with (output / "alternative_comparisons.csv").open() as handle:
            alternatives = list(csv.DictReader(handle))
        self.assertEqual(len(alternatives), 4000)
        self.assertEqual(
            sum(
                (
                    r["status"] == "unavailable_due_to_power_law_failure"
                    for r in alternatives
                )
            ),
            2,
        )
        self.assertEqual(sum((r["status"] == "failed" for r in alternatives)), 1)
        with (output / "family_summary.csv").open() as handle:
            families = list(csv.DictReader(handle))
        er = families[0]
        self.assertEqual(er["family"], "erdos_renyi")
        expected = dict(
            graphs_recorded=400,
            pl_valid_count=399,
            pl_failed_count=1,
            pl_insufficient_tail_count=1,
            pl_adequate_tail_count=398,
            pl_nominal_not_rejected_count_all_valid=399,
            pl_nominal_not_rejected_count_adequate=398,
            pl_mc_interval_crosses_01_count_all_valid=399,
            pl_mc_interval_crosses_01_count_adequate=398,
            exponential_all_valid_pl_prefer_pl_count=1,
            exponential_all_valid_pl_inconclusive_count=398,
            lognormal_all_valid_pl_valid_count=398,
            lognormal_all_valid_pl_failed_count=1,
            lognormal_all_valid_pl_prefer_alternative_count=1,
            lognormal_adequate_and_not_rejected_pl_denominator=398,
            lognormal_adequate_and_not_rejected_pl_valid_count=397,
        )
        for key, value in expected.items():
            self.assertEqual(int(er[key]), value, key)
        self.assertEqual(float(er["pl_alpha_median_all_valid"]), 2.5)
        self.assertEqual(float(er["pl_max_degree_over_xmin_median_all_valid"]), 5.0)
        for filename, info in audit["output_files"].items():
            self.assertEqual(summary.file_hash(output / filename), info["sha256"])
            with (output / filename).open() as handle:
                columns = csv.DictReader(handle).fieldnames
            self.assertEqual(set(columns), set(audit["columns"][filename]))

    def test_outer_checksum_detects_changed_result(self):
        with self.changed_checkpoint(10, lambda r: r.update(seconds=0.02)):
            with self.assertRaisesRegex(ValueError, "file checksum"):
                summary.load_completed_results(self.results)

    def test_embedded_checksum_detects_changed_payload(self):
        with self.changed_checkpoint(10, lambda r: r.update(seconds=0.02), outer=True):
            with self.assertRaisesRegex(ValueError, "Embedded checkpoint"):
                summary.load_completed_results(self.results)

    def test_same_tail_mismatch_rejected_even_with_updated_hashes(self):
        def mutate(result):
            result["alternatives"]["exponential"]["n_tail"] = 99

        with self.changed_checkpoint(10, mutate, embedded=True, outer=True):
            with self.assertRaisesRegex(ValueError, "same tail"):
                summary.load_completed_results(self.results)

    def test_interpretation_flag_mismatch_rejected(self):
        def mutate(result):
            result["power_law"]["insufficient_tail_count"] = True

        with self.changed_checkpoint(10, mutate, embedded=True, outer=True):
            with self.assertRaisesRegex(ValueError, "flags disagree"):
                summary.load_completed_results(self.results)

    def test_incomplete_study_creates_no_output(self):
        report_path = self.results / "completion_report.json"
        old = report_path.read_bytes()
        target = self.root / "must_not_exist"
        try:
            report = json.loads(old)
            report["recorded_graphs"] = 1999
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "2,000"):
                summary.summarise(self.results, target)
            self.assertFalse(target.exists())
        finally:
            report_path.write_bytes(old)

    def test_existing_output_is_never_overwritten(self):
        target = self.root / "existing"
        target.mkdir()
        (target / "keep.txt").write_text("retain")
        with self.assertRaises(FileExistsError):
            summary.summarise(self.results, target)
        self.assertEqual((target / "keep.txt").read_text(), "retain")

    def test_active_analysis_lock_blocks_summary(self):
        lock = self.results / ".analysis.lock"
        lock.write_text("ARTIFICIAL")
        try:
            with self.assertRaisesRegex(ValueError, "lock"):
                summary.load_completed_results(self.results)
        finally:
            lock.unlink()


if __name__ == "__main__":
    unittest.main()
