import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "benchmark_report", ROOT / "benchmarks/benchmark_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def document(wall=0.1, memory=100, revision="current"):
    return {
        "schema": 1,
        "date": "2026-11-13",
        "revision": revision,
        "configuration": {},
        "results": [
            {
                "case": "score",
                "threads": 1,
                "status": "ok",
                "workload": "identical-inputs",
                "wall_seconds": wall,
                "cpu_seconds": wall,
                "peak_rss_mib": memory,
                "samples": 378,
                "variants": 15938,
                "blocks": 16,
                "python": "3.12",
                "platform": "Linux",
                "machine": "x86_64",
                "cpu": "test CPU",
                "openmp": True,
            }
        ],
    }


@pytest.mark.parametrize(
    "wall,memory,count",
    [
        (0.12, 110, 0),
        (0.14, 100, 1),
        (0.1, 140, 1),
        (0.14, 140, 2),
    ],
)
def test_detects_time_and_memory_regressions(wall, memory, count):
    ratios, failures = report.compare(document(wall, memory), document(revision="base"))
    assert len(failures) == count
    assert ratios[("score", 1)][:2] == pytest.approx((wall / 0.1, memory / 100))


def test_absolute_floors_avoid_tiny_noisy_regressions():
    _, failures = report.compare(document(0.002, 11), document(0.001, 1))
    assert failures == []


@pytest.mark.parametrize("metric", ["wall_seconds", "peak_rss_mib"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_measurements_fail_closed(metric, value):
    current = document()
    current["results"][0][metric] = value
    with pytest.raises(ValueError, match="Invalid"):
        report.compare(current)


def test_different_inputs_or_hardware_are_not_compared():
    base = document()
    base["results"][0]["workload"] = "another-machine-or-input"
    with pytest.raises(ValueError, match="Incomparable"):
        report.compare(document(), base)


def test_new_case_can_establish_a_baseline_but_current_run_cannot_skip():
    base = document(revision="base")
    base["results"][0] = {"case": "score", "threads": 1, "status": "unsupported"}
    ratios, failures = report.compare(document(), base)
    assert ratios[("score", 1)] is None and failures == []
    with pytest.raises(ValueError, match="did not succeed"):
        report.compare(base)
    with pytest.raises(ValueError, match="empty"):
        report.compare({"schema": 1, "results": []})


def test_duplicate_cases_are_rejected():
    current = document()
    current["results"].append(copy.deepcopy(current["results"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        report.compare(current)


def test_history_merge_is_idempotent_after_csv_round_trip():
    history = report.merge_history([], document(revision="base"))
    history = [{key: str(value) for key, value in row.items()} for row in history]
    merged = report.merge_history(history, document())
    repeated = report.merge_history(merged, document(0.08))
    assert len(repeated) == 2
    assert {row["revision"] for row in repeated} == {"base", "current"}
    assert (
        next(row for row in repeated if row["revision"] == "current")["wall_seconds"]
        == 0.08
    )


def test_failing_comparison_still_writes_reports(tmp_path):
    current, base = tmp_path / "current.json", tmp_path / "base.json"
    current.write_text(json.dumps(document(0.2)))
    base.write_text(json.dumps(document(revision="base")))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_report.py"),
            "--current",
            str(current),
            "--baseline",
            str(base),
            "--history",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "regression" in result.stderr.lower()
    assert (tmp_path / "out/results.csv").is_file()
    assert "2.00x baseline" in (tmp_path / "out/results.md").read_text()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Benchmark RSS profiling targets Linux/macOS."
)
def test_native_benchmark_smoke_all_cases(tmp_path):
    pytest.importorskip("magenpy.stats.variant.variant_cpp")
    import magenpy as mgp

    installed_root = Path(mgp.__file__).resolve().parent.parent
    output = tmp_path / "measurements.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_native.py"),
            "--source-root",
            str(installed_root),
            "--bfile",
            mgp.tgp_eur_data_path(),
            "--revision",
            "a" * 40,
            "--push-date",
            "1700000000",
            "--output",
            str(output),
            "--threads",
            "1",
            "--snps",
            "128",
            "--samples",
            "48",
            "--block-snps",
            "32",
            "--window-size",
            "16",
            "--repeats",
            "1",
            "--min-seconds",
            ".01",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(output.read_text())
    assert len(report.successful_results(measured)) == 8
    assert measured["revision"] == "a" * 40
    assert measured["push_date"] == "2023-11-14T22:13:20+00:00"
    assert "date" in measured and isinstance(measured["dirty"], bool)
    assert len({row["workload"] for row in measured["results"]}) == 1
    assert all(row["calls_per_repeat"] >= 1 for row in measured["results"])
    assert all(len(row["wall_samples"]) == 1 for row in measured["results"])


@pytest.mark.skipif(
    sys.platform == "win32", reason="Benchmark RSS profiling targets Linux/macOS."
)
def test_worker_rejects_imports_from_the_wrong_checkout(tmp_path):
    pytest.importorskip("magenpy.stats.variant.variant_cpp")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_native.py"),
            "--worker",
            "score",
            "--source-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "expected checkout" in result.stderr


def test_grouped_history_links_commits_and_retains_push_and_run_dates():
    current, older = document(revision="a" * 40), document(revision="b" * 40)
    current.update(
        date="2026-11-13T12:00:00+00:00", push_date="2026-11-12T23:59:00+00:00"
    )
    older.update(
        date="2026-11-11T12:00:00+00:00", push_date="2026-11-10T12:00:00+00:00"
    )
    for item in (current, older):
        template = item["results"][0]
        item["results"] = [
            dict(template, case=case)
            for case in ("ld_block", "score", "gwas", "variant_stats")
        ]
    comparisons, failures = report.compare(current)
    rows = report.merge_history(report.merge_history([], older), current, comparisons)
    text = report.markdown(current, comparisons, failures, rows, "owner/magenpy")
    for category in ("LD calculation", "Scoring", "GWAS", "Variant statistics"):
        assert f"## {category}\n" in text
    for function in (
        "Block LD (native single pass)",
        "Single score",
        "Exclude missing calls per SNP",
        "Allele frequencies and sample counts",
    ):
        section = text.split(f"### {function}\n", 1)[1].split("\n##", 1)[0]
        assert section.index("a" * 40) < section.index("b" * 40)
        assert "2026-11-12 | 2026-11-13" in section
        assert "https://github.com/owner/magenpy/commit/" + "a" * 40 in section
    assert len(rows) == 8


def test_modified_checkout_is_distinct_from_clean_commit():
    clean = document(revision="a" * 40)
    dirty = dict(copy.deepcopy(clean), dirty=True)
    rows = report.merge_history(report.merge_history([], clean), dirty)
    assert len(rows) == 2
    assert report.commit_link(dirty, "shz9/magenpy").endswith("+ local changes")


def test_legacy_history_is_migrated_without_inventing_push_dates():
    legacy = {
        **document(revision="b" * 40)["results"][0],
        "date": "2026-11-11",
        "revision": "b" * 40,
    }
    rows = report.merge_history([legacy], document(revision="a" * 40))
    prior = next(row for row in rows if row["revision"] == "b" * 40)
    assert prior["category"] == "Scoring" and prior["function"] == "Single score"
    assert prior["push_date"] == "" and prior["dirty"] is False


def test_comparison_ratios_survive_in_history():
    current, base = document(0.14, 140, "a" * 40), document(revision="b" * 40)
    ratios, _ = report.compare(current, base)
    rows = report.merge_history([], current, ratios, base["revision"])
    latest = rows[0]
    assert latest["time_ratio"] == pytest.approx(1.4)
    assert latest["memory_ratio"] == pytest.approx(1.4)
    assert latest["baseline_revision"] == base["revision"]
    converted = [{key: str(value) for key, value in row.items()} for row in rows]
    updated = document(revision="c" * 40)
    text = report.markdown(
        updated,
        report.compare(updated)[0],
        [],
        report.merge_history(converted, updated),
    )
    assert "1.40x | 1.40x" in text
