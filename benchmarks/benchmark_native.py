"""Measure native genotype workloads in fresh processes using the bundled BED panel."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

CASES = (
    "variant_stats",
    "score",
    "score_standardized",
    "score_batched",
    "gwas",
    "gwas_imputed",
    "ld_windowed",
    "ld_block",
)
BLOCK_COMPARISONS = ("ld_block_fallback", "ld_block_plink")
ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity(root, requested_revision=None):
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top).resolve() != root.resolve():
            return requested_revision or "unversioned", False
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        changed = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                "magenpy",
                "setup.py",
                "pyproject.toml",
                "benchmarks",
            ],
            text=True,
        ).strip()
        return requested_revision or revision, bool(changed)
    except (OSError, subprocess.CalledProcessError):
        return requested_revision or "unversioned", False


def normalize_push_date(value):
    if not value:
        return ""
    try:
        result = datetime.fromtimestamp(float(value), timezone.utc)
    except ValueError:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).isoformat()


def cpu_name():
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor()


def peak_rss_mib(usage):
    return usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


def worker(args):
    import numpy as np
    import magenpy as mgp
    from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix, plinkBEDGenotypeMatrix
    from magenpy.stats.ld.c_utils import openmp_enabled
    from magenpy.stats.ld.estimator import BlockLD
    from magenpy.stats.gwa.utils import perform_gwa_magenpy

    if (
        args.source_root
        and Path(mgp.__file__).resolve().parent.parent != args.source_root.resolve()
    ):
        raise RuntimeError(
            f"Imported {mgp.__file__}, expected checkout {args.source_root}."
        )
    if args.require_openmp and not openmp_enabled():
        raise RuntimeError("Continuous benchmarks require an OpenMP-enabled build.")
    if (
        args.worker == "gwas_imputed"
        and "impute_missing" not in inspect.signature(perform_gwa_magenpy).parameters
    ):
        if args.allow_unsupported:
            return {
                "case": args.worker,
                "threads": args.threads[0],
                "status": "unsupported",
            }
        raise RuntimeError(
            "The current native backend must support GWAS mean imputation."
        )
    threads = args.threads[0]
    with tempfile.TemporaryDirectory(prefix="magenpy-benchmark-") as workspace:
        g = MagenpyGenotypeMatrix.from_file(
            args.bfile, temp_dir=workspace, threads=threads
        )
        if args.samples:
            g.filter_samples(keep_samples=g.samples[: args.samples])
        if args.snps:
            g.filter_snps(extract_snps=g.snps[: args.snps])
        g.drop_duplicated_snps()
        g.filter_by_allele_frequency(min_mac=1)
        if g.sample_size < 3 or g.n_snps < 3 or len(g.chromosomes) != 1:
            raise ValueError(
                "Benchmark inputs must contain >=3 samples/SNPs on one chromosome."
            )
        rng = np.random.default_rng(319)
        beta = rng.normal(0, 0.01, size=g.n_snps)
        beta_matrix = rng.normal(0, 0.01, size=(g.n_snps, 16))
        phenotype = rng.normal(size=g.sample_size)
        g.sample_table.set_phenotype(phenotype, phenotype_likelihood="gaussian")
        # Score frequencies and LD inputs are warm in both revisions. Variant
        # statistics explicitly recompute their outputs on every timed call.
        _ = g.maf, g.n_per_snp
        positions = g.bp_pos
        starts = np.unique(positions[:: args.block_snps]).astype(np.int64)
        blocks = np.column_stack((starts, np.r_[starts[1:], int(positions[-1]) + 1]))
        expected_snps = g.snps.copy()
        selection = fingerprint(
            {
                "samples": g.sample_index.tolist(),
                "variants": g.snp_index.tolist(),
                "snps": expected_snps.tolist(),
            }
        )
        if args.worker == "ld_block_plink":
            g = plinkBEDGenotypeMatrix(
                sample_table=g.sample_table,
                snp_table=g.snp_table,
                bed_file=args.bfile,
                temp_dir=workspace,
                threads=threads,
            )
        is_ld = args.worker.startswith("ld_")

        def operation(output):
            if args.worker == "variant_stats":
                g.compute_allele_frequency()
                return g.maf
            if args.worker.startswith("score"):
                return g.score(
                    beta_matrix if args.worker == "score_batched" else beta,
                    standardize_genotype=args.worker == "score_standardized",
                    impute_missing=True,
                )
            if args.worker.startswith("gwas"):
                kwargs = (
                    {"impute_missing": True} if args.worker == "gwas_imputed" else {}
                )
                return g.perform_gwas(**kwargs)
            if args.worker == "ld_block_fallback":
                return BlockLD(g, ld_blocks=blocks)._compute_blockwise(
                    output,
                    overwrite=True,
                    delete_original=True,
                    dtype=args.dtype,
                    compressor_name="zstd",
                    compression_level=7,
                    compute_spectral_properties=False,
                    store_type=None,
                )
            options = (
                {"window_size": args.window_size}
                if args.worker == "ld_windowed"
                else {"ld_blocks": blocks}
            )
            return g.compute_ld(
                "windowed" if args.worker == "ld_windowed" else "block",
                output,
                dtype=args.dtype,
                **options,
            )

        def validate(result):
            if is_ld:
                assert result is not None and result.validate_ld_matrix()
                np.testing.assert_array_equal(result.snps, expected_snps)
                assert result.sample_size == g.sample_size and result.data.size > 0
                assert np.isfinite(result.data[:1024]).all()
            elif args.worker.startswith("gwas"):
                assert len(result.table) == g.n_snps
                assert np.isfinite(result.beta_hat).all()
                assert np.isfinite(result.se).all()
            else:
                expected_shape = (
                    (g.sample_size, 16)
                    if args.worker == "score_batched"
                    else (g.sample_size,)
                    if args.worker.startswith("score")
                    else (g.n_snps,)
                )
                assert result.shape == expected_shape and np.isfinite(result).all()

        samples, cpu_samples, calls = [], [], 1
        for repetition in range(args.repeats + 1):
            # Temp directory creation, validation, and removal are outside timing.
            # LD timing includes estimator setup, BED reads, LD scores, compression,
            # validation internal to compute_ld, and writing the complete Zarr store.
            with tempfile.TemporaryDirectory(dir=workspace) as output:
                before = os.times()
                start = perf_counter()
                for _ in range(calls):
                    result = operation(output)
                elapsed = (perf_counter() - start) / calls
                after = os.times()
                cpu = sum(after[:4]) - sum(before[:4])
                validate(result)
                del result
            if repetition:
                samples.append(elapsed)
                cpu_samples.append(cpu / calls)
            elif not is_ld:
                calls = min(
                    1000, max(1, math.ceil(args.min_seconds / max(elapsed, 1e-9)))
                )
        packages = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "zarr", "numcodecs", "joblib")
        }
        return {
            "case": args.worker,
            "threads": threads,
            "status": "ok",
            "samples": g.sample_size,
            "variants": g.n_snps,
            "blocks": len(blocks),
            "selection": selection,
            "wall_seconds": statistics.median(samples),
            "wall_samples": samples,
            "cpu_seconds": statistics.median(cpu_samples),
            "calls_per_repeat": calls,
            "peak_rss_mib": peak_rss_mib(resource.getrusage(resource.RUSAGE_SELF)),
            "child_peak_rss_mib": peak_rss_mib(
                resource.getrusage(resource.RUSAGE_CHILDREN)
            ),
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu": cpu_name(),
            "openmp": openmp_enabled(),
            "packages": packages,
            "source": str(Path(mgp.__file__).resolve()),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfile", default=str(ROOT / "magenpy/data/1000G_eur_chr22"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-root", type=Path, help="Require imports from this exact checkout."
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-seconds", type=float, default=0.15)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--block-snps", type=int, default=1000)
    parser.add_argument(
        "--dtype", choices=["int8", "int16", "float32", "float64"], default="int16"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASES + BLOCK_COMPARISONS, default=CASES
    )
    parser.add_argument(
        "--samples", type=int, default=0, help="Optional sample limit for smoke tests."
    )
    parser.add_argument(
        "--snps", type=int, default=0, help="Optional SNP limit for smoke tests."
    )
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument(
        "--push-date",
        default=os.environ.get("BENCHMARK_PUSH_DATE", ""),
        help="Push-event timestamp (ISO 8601 or Unix seconds); leave empty for local/PR/manual runs.",
    )
    parser.add_argument("--require-openmp", action="store_true")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Baseline only: record newly introduced cases as unavailable.",
    )
    parser.add_argument(
        "--worker", choices=CASES + BLOCK_COMPARISONS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if (
        min(*args.threads, args.repeats, args.window_size, args.block_snps) < 1
        or not math.isfinite(args.min_seconds)
        or args.min_seconds <= 0
        or min(args.samples, args.snps) < 0
    ):
        parser.error(
            "Counts and durations must be positive; optional input limits must be nonnegative."
        )
    args.bfile = str(Path(args.bfile).resolve()).removesuffix(".bed")
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required.")
    data_hash = hashlib.sha256()
    for suffix in (".bed", ".bim", ".fam"):
        with Path(args.bfile + suffix).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data_hash.update(chunk)
    configuration = {
        key: getattr(args, key)
        for key in (
            "window_size",
            "block_snps",
            "dtype",
            "samples",
            "snps",
            "repeats",
            "min_seconds",
        )
    }
    revision, dirty = source_identity(args.source_root or ROOT, args.revision)
    document = {
        "schema": 1,
        "revision": revision,
        "dirty": dirty,
        "push_date": normalize_push_date(args.push_date),
        "date": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": data_hash.hexdigest(),
        "configuration": configuration,
        "results": [],
    }
    for threads in args.threads:
        for case in args.cases:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                case,
                "--threads",
                str(threads),
                "--bfile",
                args.bfile,
            ]
            if args.source_root:
                command.extend(["--source-root", str(args.source_root.resolve())])
            for name, value in configuration.items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            for flag in ("require_openmp", "allow_unsupported"):
                if getattr(args, flag):
                    command.append("--" + flag.replace("_", "-"))
            env = dict(os.environ)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = "1"
            env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
            completed = subprocess.run(command, text=True, capture_output=True, env=env)
            if completed.returncode:
                raise RuntimeError(
                    f"{case} ({threads} threads) failed:\n{completed.stdout}\n{completed.stderr}"
                )
            row = json.loads(completed.stdout)
            if row["status"] == "ok":
                identity = {
                    key: row[key]
                    for key in (
                        "selection",
                        "python",
                        "platform",
                        "machine",
                        "cpu",
                        "openmp",
                        "packages",
                    )
                }
                row["workload"] = fingerprint(
                    {
                        **identity,
                        "dataset": document["dataset_sha256"],
                        "configuration": configuration,
                    }
                )
                print(
                    f"{case:22} threads={threads} {row['wall_seconds']:.6f}s peak={row['peak_rss_mib']:.1f} MiB",
                    flush=True,
                )
            document["results"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(document, indent=2, allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
