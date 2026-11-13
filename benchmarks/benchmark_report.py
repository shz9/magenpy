"""Compare matched native benchmark runs and maintain CSV/Markdown history."""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

FIELDS = (
    "date",
    "push_date",
    "revision",
    "dirty",
    "category",
    "function",
    "case",
    "threads",
    "workload",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_mib",
    "samples",
    "variants",
    "blocks",
    "python",
    "platform",
    "machine",
    "cpu",
    "openmp",
    "baseline_revision",
    "time_ratio",
    "memory_ratio",
)


CASE_LABELS = {
    "ld_windowed": ("LD calculation", "Windowed LD"),
    "ld_block": ("LD calculation", "Block LD (native single pass)"),
    "ld_block_fallback": ("LD calculation", "Block LD (native per-block fallback)"),
    "ld_block_plink": ("LD calculation", "Block LD (PLINK 1.9)"),
    "score": ("Scoring", "Single score"),
    "score_standardized": ("Scoring", "Standardized score"),
    "score_batched": ("Scoring", "16-score batch"),
    "gwas": ("GWAS", "Exclude missing calls per SNP"),
    "gwas_imputed": ("GWAS", "Mean imputation"),
    "variant_stats": ("Variant statistics", "Allele frequencies and sample counts"),
}


def successful_results(document, allow_unsupported=False):
    results, seen = {}, set()
    for result in document["results"]:
        key = result["case"], int(result["threads"])
        if key in seen:
            raise ValueError(f"Duplicate benchmark case: {key}")
        seen.add(key)
        if result["status"] == "unsupported" and allow_unsupported:
            continue
        if result["status"] != "ok":
            raise ValueError(f"Benchmark did not succeed: {key}")
        for metric in ("wall_seconds", "peak_rss_mib"):
            if not math.isfinite(result[metric]) or result[metric] <= 0:
                raise ValueError(f"Invalid {metric} for {key}")
        results[key] = result
    if not results and not allow_unsupported:
        raise ValueError("Cannot report an empty benchmark run.")
    return results


def compare(
    current,
    baseline=None,
    time_ratio=1.25,
    memory_ratio=1.25,
    min_seconds=0.005,
    min_mib=16,
):
    rows = successful_results(current)
    previous = successful_results(baseline, allow_unsupported=True) if baseline else {}
    failures, comparisons = [], {}
    for key, row in rows.items():
        base = previous.get(key)
        if base is None:
            comparisons[key] = None
            continue
        if row["workload"] != base["workload"]:
            raise ValueError(
                f"Incomparable workload/environment for {key}; do not compare different inputs or machines."
            )
        wall = row["wall_seconds"] / base["wall_seconds"]
        memory = row["peak_rss_mib"] / base["peak_rss_mib"]
        comparisons[key] = wall, memory, base["wall_seconds"]
        if (
            wall > time_ratio
            and row["wall_seconds"] - base["wall_seconds"] > min_seconds
        ):
            failures.append(
                f"{key[0]} ({key[1]} threads): runtime {wall:.2f}x baseline"
            )
        if (
            memory > memory_ratio
            and row["peak_rss_mib"] - base["peak_rss_mib"] > min_mib
        ):
            failures.append(
                f"{key[0]} ({key[1]} threads): peak RSS {memory:.2f}x baseline"
            )
    return comparisons, failures


def history_key(row):
    # A local modified checkout is a different observation from its clean commit.
    return (
        row["revision"],
        row["case"],
        str(row["threads"]),
        row["workload"],
        str(row.get("dirty", False)).lower(),
    )


def merge_history(history, current, comparisons=None, baseline_revision=""):
    successful_results(current)
    records = {}
    for previous in history:
        row = {key: previous.get(key, "") for key in FIELDS}
        row["dirty"] = previous.get("dirty", False)
        row["category"], row["function"] = CASE_LABELS.get(
            row["case"], ("Other", row["case"])
        )
        records[history_key(row)] = row
    for result in current["results"]:
        row = {key: result.get(key, current.get(key, "")) for key in FIELDS}
        row["dirty"] = current.get("dirty", False)
        row["category"], row["function"] = CASE_LABELS.get(
            row["case"], ("Other", row["case"])
        )
        row["baseline_revision"] = baseline_revision
        comparison = (comparisons or {}).get((row["case"], int(row["threads"])))
        if comparison:
            row["time_ratio"], row["memory_ratio"] = comparison[:2]
        records[history_key(row)] = row
    return sorted(
        records.values(),
        key=lambda row: (row["date"], row["case"], int(row["threads"])),
    )


def commit_link(row, repository):
    revision = row["revision"]
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
        text = f"[`{revision[:7]}`](https://github.com/{repository}/commit/{revision})"
    else:
        text = f"`{revision}`"
    if str(row.get("dirty", False)).lower() == "true":
        text += " + local changes"
    return text


def markdown(current, comparisons, failures, history=None, repository="shz9/magenpy"):
    rows = successful_results(current)
    example = next(iter(rows.values()))
    history = (
        history if history is not None else merge_history([], current, comparisons)
    )
    lines = [
        "# Magenpy continuous benchmark results",
        "",
        f"Latest revision: {commit_link(current, repository)}. "
        f"Push date (UTC): {current.get('push_date') or 'not recorded (local/PR/manual run)'}. "
        f"Benchmark run (UTC): {current['date']}.",
        "",
        f"Latest host: {example['cpu']} ({example['platform']}/{example['machine']}); "
        f"Python {example['python']}; OpenMP: {example['openmp']}.",
        f"Panel: {example['samples']:,} samples, {example['variants']:,} variants, "
        f"{example['blocks']} artificial blocks. Configuration: `{json.dumps(current['configuration'], sort_keys=True)}`.",
        "",
        "Each function has its own history table, newest runs first. Commit links "
        "identify the measured source; '+ local changes' marks an uncommitted checkout. "
        "Push dates come from push-event metadata; a dash means no push date was recorded. "
        "Run dates are recorded separately for local, PR, and manual benchmarks.",
        "",
        "Times are medians after one warm-up in fresh worker processes. LD includes "
        "complete Zarr output; temporary-file cleanup is excluded. RSS is the worker's "
        "lifetime high-water mark including setup and validation, excluding PLINK children.",
        "",
    ]
    categories = dict.fromkeys(category for category, _ in CASE_LABELS.values())
    categories.update(dict.fromkeys(row["category"] for row in history))
    for category in categories:
        grouped = [row for row in history if row["category"] == category]
        if not grouped:
            continue
        lines += [f"## {category}", ""]
        cases = dict.fromkeys(
            case for case, labels in CASE_LABELS.items() if labels[0] == category
        )
        cases.update(dict.fromkeys(row["case"] for row in grouped))
        for case in cases:
            records = [row for row in grouped if row["case"] == case]
            if not records:
                continue
            records.sort(
                key=lambda row: (row["date"], -int(row["threads"])), reverse=True
            )
            lines += [
                f"### {records[0]['function']}",
                "",
                "| Commit ID | Push date (UTC) | Run date (UTC) | Threads | Runtime (ms) | Peak RSS (MiB) | Time / base | RSS / base | Host |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
            for row in records:
                wall = (
                    f"{float(row['time_ratio']):.2f}x"
                    if row.get("time_ratio") not in (None, "")
                    else ("new" if row.get("baseline_revision") else "—")
                )
                memory = (
                    f"{float(row['memory_ratio']):.2f}x"
                    if row.get("memory_ratio") not in (None, "")
                    else "—"
                )
                host = f"{row['cpu']} ({row['platform']}/{row['machine']}; OMP {row['openmp']})"
                lines.append(
                    f"| {commit_link(row, repository)} | {str(row.get('push_date') or '—')[:10]} | "
                    f"{row['date'][:10]} | {row['threads']} | {float(row['wall_seconds']) * 1000:.3f} | "
                    f"{float(row['peak_rss_mib']):.1f} | {wall} | {memory} | {host} |"
                )
            lines.append("")
    lines += ["## Latest regression check", ""]
    lines += (
        ["- " + message for message in failures]
        if failures
        else [
            "No detected regressions in comparable cases. New cases establish a baseline."
        ]
    )
    lines += [
        "",
        "[results.csv](results.csv) retains full commit IDs, push/run timestamps, "
        "function/category labels, comparison ratios, and workload fingerprints. "
        "Raw run artifacts also retain individual timing samples and dependency versions. "
        "Compare matching workload fingerprints; ratios above 1 indicate increased time or memory.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "shz9/magenpy")
    )
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--history", type=Path, default=Path("benchmarks/results.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-ratio", type=float, default=1.25)
    parser.add_argument("--memory-ratio", type=float, default=1.25)
    parser.add_argument("--min-seconds", type=float, default=0.005)
    parser.add_argument("--min-mib", type=float, default=16)
    args = parser.parse_args(argv)
    if (
        not all(
            math.isfinite(value)
            for value in (
                args.time_ratio,
                args.memory_ratio,
                args.min_seconds,
                args.min_mib,
            )
        )
        or min(args.time_ratio, args.memory_ratio) < 1
        or min(args.min_seconds, args.min_mib) < 0
    ):
        parser.error(
            "Ratios must be finite and >=1; minimum increases must be finite and nonnegative."
        )
    current = json.loads(args.current.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    comparisons, failures = compare(
        current,
        baseline,
        args.time_ratio,
        args.memory_ratio,
        args.min_seconds,
        args.min_mib,
    )
    history = []
    if args.history.exists():
        with args.history.open(newline="") as stream:
            history = list(csv.DictReader(stream))
    rows = merge_history(
        history, current, comparisons, baseline["revision"] if baseline else ""
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "results.md").write_text(
        markdown(current, comparisons, failures, rows, args.repository)
    )
    if failures:
        raise SystemExit("Performance regression detected:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
