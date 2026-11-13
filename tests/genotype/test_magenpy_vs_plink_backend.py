import importlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import magenpy as mgp
from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix, plinkBEDGenotypeMatrix


pytestmark = pytest.mark.plink

OUTPUT_ATOL = 1e-5
SCORE_ATOL = 1e-4


def _unavailable(message):
    if os.environ.get("MAGENPY_REQUIRE_PLINK") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def _require_native_extensions():
    required = {
        "magenpy.stats.variant.variant_cpp": (
            "extract_genotype_matrix",
            "compute_variant_stats",
            "compute_gwa_linear_stats",
        ),
        "magenpy.stats.score.score_cpp": ("calculate_pgs",),
        "magenpy.stats.ld.c_utils": ("compute_ut_ld_from_bed",),
    }
    for module_name, functions in required.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            _unavailable(f"Cannot import {module_name}: {error}")
        for function in functions:
            if not hasattr(module, function):
                _unavailable(
                    f"Rebuild native extensions: {module_name}.{function} is missing."
                )


def _require_plink():
    for executable in ("plink", "plink2"):
        if shutil.which(executable) is None:
            _unavailable(f"{executable} executable is not available on PATH.")


@pytest.fixture(scope="module", autouse=True)
def require_comparison_backends():
    _require_native_extensions()
    _require_plink()


@pytest.fixture(scope="module")
def backend_comparison_inputs(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("backend_discovery")
    discovery = MagenpyGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(temp_dir),
        threads=2,
    )
    discovery.filter_samples(keep_samples=discovery.samples[:96])
    discovery.filter_snps(extract_snps=discovery.snps[:1024])

    genotype = discovery.to_numpy(dtype=np.int8)
    complete_mask = np.all(genotype >= 0, axis=0)
    complete_snps = discovery.snps[complete_mask]

    if complete_snps.shape[0] < 32:
        pytest.fail(
            "Packaged reference data must contain at least 32 complete variants."
        )

    inputs = {
        "samples": discovery.samples.copy(),
        "stat_snps": discovery.snps[:96].copy(),
        "complete_snps": complete_snps[:48].copy(),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)
    return inputs


def _make_backend_pair(tmp_path, inputs, snps):
    plink_temp = tmp_path / "plink_tmp"
    magenpy_temp = tmp_path / "magenpy_tmp"
    plink_temp.mkdir()
    magenpy_temp.mkdir()

    plink_gmat = plinkBEDGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(plink_temp),
        threads=2,
    )
    magenpy_gmat = MagenpyGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(magenpy_temp),
        threads=2,
    )

    for gmat in (plink_gmat, magenpy_gmat):
        gmat.filter_samples(keep_samples=inputs["samples"])
        gmat.filter_snps(extract_snps=snps)

        sample_idx = np.arange(gmat.sample_size, dtype=np.float64)
        phenotype = (
            0.03 * sample_idx + np.sin(sample_idx / 5.0) + np.cos(sample_idx / 11.0)
        )
        gmat.sample_table.set_phenotype(phenotype, phenotype_likelihood="gaussian")

    return plink_gmat, magenpy_gmat


def _align_sumstats(plink_sumstats, magenpy_sumstats):
    columns = ["SNP", "MAF", "N", "BETA", "SE", "Z", "PVAL"]
    plink_table = plink_sumstats.to_table(columns)
    magenpy_table = magenpy_sumstats.to_table(columns)

    merged = plink_table.merge(
        magenpy_table,
        on="SNP",
        suffixes=("_plink", "_magenpy"),
    )
    assert len(merged) == len(magenpy_table) == len(plink_table)
    return merged


def test_variant_statistics_match_plink2(tmp_path, backend_comparison_inputs):
    plink_gmat, magenpy_gmat = _make_backend_pair(
        tmp_path,
        backend_comparison_inputs,
        backend_comparison_inputs["stat_snps"],
    )

    plink_gmat.compute_allele_frequency()
    plink_gmat.compute_sample_size_per_snp()
    magenpy_gmat.compute_allele_frequency()
    magenpy_gmat.compute_sample_size_per_snp()

    pd.testing.assert_index_equal(
        pd.Index(plink_gmat.snps),
        pd.Index(magenpy_gmat.snps),
    )
    np.testing.assert_array_equal(magenpy_gmat.n_per_snp, plink_gmat.n_per_snp)
    np.testing.assert_allclose(
        magenpy_gmat.maf,
        plink_gmat.maf,
        rtol=0.0,
        atol=OUTPUT_ATOL,
        equal_nan=True,
    )


@pytest.mark.parametrize("standardize_genotype", [False, True])
def test_score_matches_plink2(
    tmp_path, backend_comparison_inputs, standardize_genotype
):
    plink_gmat, magenpy_gmat = _make_backend_pair(
        tmp_path,
        backend_comparison_inputs,
        backend_comparison_inputs["complete_snps"],
    )

    beta = np.column_stack(
        [
            np.linspace(-0.15, 0.2, magenpy_gmat.n_snps),
            np.cos(np.arange(magenpy_gmat.n_snps) / 7.0),
        ]
    )

    plink_score = plink_gmat.score(beta, standardize_genotype=standardize_genotype)
    magenpy_score = magenpy_gmat.score(beta, standardize_genotype=standardize_genotype)

    np.testing.assert_allclose(
        magenpy_score,
        plink_score,
        rtol=0.0,
        atol=SCORE_ATOL,
    )


def test_gwas_matches_plink2_on_complete_variants(tmp_path, backend_comparison_inputs):
    plink_gmat, magenpy_gmat = _make_backend_pair(
        tmp_path,
        backend_comparison_inputs,
        backend_comparison_inputs["complete_snps"],
    )

    plink_sumstats = plink_gmat.perform_gwas()
    magenpy_sumstats = magenpy_gmat.perform_gwas()
    merged = _align_sumstats(plink_sumstats, magenpy_sumstats)

    np.testing.assert_array_equal(merged["N_magenpy"].values, merged["N_plink"].values)
    np.testing.assert_allclose(
        merged["MAF_magenpy"].values,
        merged["MAF_plink"].values,
        rtol=0.0,
        atol=OUTPUT_ATOL,
    )
    np.testing.assert_allclose(
        merged["BETA_magenpy"].values,
        merged["BETA_plink"].values,
        rtol=0.0,
        atol=OUTPUT_ATOL,
    )
    np.testing.assert_allclose(
        merged["SE_magenpy"].values,
        merged["SE_plink"].values,
        rtol=0.0,
        atol=OUTPUT_ATOL,
    )


def test_windowed_ld_matches_plink1p9_on_complete_variants(
    tmp_path, backend_comparison_inputs
):
    snps = backend_comparison_inputs["complete_snps"][:32]
    plink_gmat, magenpy_gmat = _make_backend_pair(
        tmp_path,
        backend_comparison_inputs,
        snps,
    )

    plink_ld = plink_gmat.compute_ld(
        "windowed",
        str(tmp_path / "plink_ld"),
        dtype="float32",
        window_size=8,
    )
    magenpy_ld = magenpy_gmat.compute_ld(
        "windowed",
        str(tmp_path / "magenpy_ld"),
        dtype="float32",
        window_size=8,
    )

    assert plink_ld.validate_ld_matrix()
    assert magenpy_ld.validate_ld_matrix()
    assert np.array_equal(plink_ld.snps, magenpy_ld.snps)

    plink_csr = plink_ld.to_csr(return_symmetric=True, dtype=np.float32)
    magenpy_csr = magenpy_ld.to_csr(return_symmetric=True, dtype=np.float32)

    np.testing.assert_allclose(
        magenpy_csr.toarray(),
        plink_csr.toarray(),
        rtol=0.0,
        atol=OUTPUT_ATOL,
    )


# These fixtures are encoded directly from known dosages, without using either
# backend to select/interpret the reference genotypes. Padding, allele direction,
# cohort differences, missingness, and perfect +/- LD are deliberately controlled.
@pytest.fixture(scope="module")
def synthetic_panel(tmp_path_factory, request):
    n = getattr(request, "param", 65)
    root = tmp_path_factory.mktemp(f"plink_panel_{n}")
    rng = np.random.default_rng(319)
    x = rng.integers(0, 3, size=(n, 48), dtype=np.int8)
    x[:24, 0], x[24:48, 0] = 0, 2
    x[:24, 1], x[24:48, 1] = 2, 0
    x[:, 2], x[:, 3], x[:, 5] = -1, 0, 2
    x[:, 6:8] = -1
    x[0, 6] = 1  # Only one observed call.
    x[:2, 7] = [0, 2]  # Two observed calls with nonzero genotype variance.
    x[:24, 4] = 0
    x[0, 4] = 1  # Exact MAC=1 boundary in the first cohort.
    x[:, 9], x[:, 10] = x[:, 8], 2 - x[:, 8]
    missing = x.copy()
    missing[:, 12:][rng.random((n, 36)) < 0.12] = -1
    common = x.copy()
    common[np.arange(n) % 9 == 0, 8:24] = -1
    samples = np.array([f"s{i:03d}" for i in range(n)])
    snps = np.array([f"rs{i}" for i in range(48)])
    panels = {}
    lookup = {-1: 1, 0: 3, 1: 2, 2: 0}
    for name, values in [("complete", x), ("missing", missing), ("common", common)]:
        prefix = root / name
        prefix.with_suffix(".fam").write_text(
            "".join(f"F {sample} 0 0 1 -9\n" for sample in samples)
        )
        prefix.with_suffix(".bim").write_text(
            "".join(
                f"{1 + j // 24}\t{snp}\t0\t{1000 + 100 * j}\tA\tG\n"
                for j, snp in enumerate(snps)
            )
        )
        encoded = bytearray(b"\x6c\x1b\x01")
        for column in values.T:
            for start in range(0, n, 4):
                encoded.append(
                    sum(
                        lookup[int(v)] << (2 * k)
                        for k, v in enumerate(column[start : start + 4])
                    )
                )
        prefix.with_suffix(".bed").write_bytes(encoded)
        panels[name] = str(prefix), values
    return {"panels": panels, "samples": samples, "snps": snps, "n": n}


def _rows(panel, selection):
    if selection == "prefix":
        return np.arange(24)
    if selection == "interior":
        return np.arange(24, 48)
    if selection == "strided":
        return np.arange(1, panel["n"] - 1, 2)
    if selection == "last":
        return np.arange(panel["n"] - 24, panel["n"])
    raise ValueError(selection)


def _statistics(x):
    n = (x >= 0).sum(axis=0)
    af = np.divide(
        np.maximum(x, 0).sum(axis=0),
        2 * n,
        out=np.full(x.shape[1], np.nan),
        where=n > 0,
    )
    return af, n


def _assert_statistics(g, x):
    af, n = _statistics(x)
    np.testing.assert_array_equal(g.n_per_snp, n)
    np.testing.assert_allclose(g.maf, af, atol=OUTPUT_ATOL, rtol=0, equal_nan=True)


def _panel_pair(tmp_path, panel, kind="complete", rows=None, cols=None, threads=2):
    prefix, x = panel["panels"][kind]
    pair = []
    for cls in (plinkBEDGenotypeMatrix, MagenpyGenotypeMatrix):
        g = cls.from_file(
            prefix, temp_dir=str(tmp_path / cls.__name__), threads=threads
        )
        # Exercise cached pooled frequencies, rather than only cold properties.
        _ = g.maf, g.n_per_snp
        if rows is not None:
            g.filter_samples(keep_samples=panel["samples"][rows])
        if cols is not None:
            # Requests may be unordered/repeated; BED order must be retained.
            requested = list(panel["snps"][cols][::-1])
            g.filter_snps(extract_snps=requested + requested[:1])
        pair.append(g)
    return pair


def _run_plink(executable, *args):
    command = [shutil.which(executable), *map(str, args), "--threads", "2"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"PLINK command failed: {command}\n{result.stdout}\n{result.stderr}"
    )
    return result


@pytest.mark.parametrize("synthetic_panel", [65, 66, 67, 68], indirect=True)
@pytest.mark.parametrize("selection", ["prefix", "interior", "strided", "last"])
def test_repeated_subset_statistics_match_plink_and_known_dosages(
    tmp_path, synthetic_panel, selection
):
    rows = _rows(synthetic_panel, selection)
    cols = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 19, 24, 31, 47])
    pair = _panel_pair(tmp_path, synthetic_panel, "missing", rows, cols)
    x = synthetic_panel["panels"]["missing"][1]
    for g in pair:
        _assert_statistics(g, x[np.ix_(rows, cols)])
        g.filter_samples(keep_samples=synthetic_panel["samples"][rows[::2]])
        _assert_statistics(g, x[np.ix_(rows[::2], cols)])
    native = pair[1]
    assert native.total_samples == synthetic_panel["n"]
    np.testing.assert_array_equal(native.to_numpy(), x[np.ix_(rows[::2], cols)])
    if selection == "prefix":
        assert native.maf[0] == 0 and native.maf[1] == 1
    elif selection == "interior":
        assert native.maf[0] == 1 and native.maf[1] == 0


@pytest.mark.parametrize("selection", ["prefix", "last"])
@pytest.mark.parametrize("split", ["chromosome", "variants", "nested", "constructor"])
@pytest.mark.parametrize("threads", [1, 2])
def test_split_offsets_and_statistics_match_plink(
    tmp_path, synthetic_panel, selection, split, threads
):
    rows = _rows(synthetic_panel, selection)
    pair = _panel_pair(tmp_path, synthetic_panel, "missing", rows, threads=threads)
    x = synthetic_panel["panels"]["missing"][1]
    children = []
    for g in pair:
        if split == "chromosome":
            groups = g.split_by_chromosome()
        elif split == "variants":
            groups = g.split_by_variants({"block": g.snps[1::3]})
        elif split == "nested":
            child = g.split_by_chromosome()[2]
            groups = child.split_by_variants({"block": child.snps[::2]})
        else:
            groups = {
                "copy": g.__class__(
                    sample_table=g.sample_table,
                    snp_table=g.snp_table,
                    bed_file=g.bed_file,
                    temp_dir=g.temp_dir,
                    threads=threads,
                )
            }
        children.append(groups)
    for key, native in children[1].items():
        plink = children[0][key]
        expected = x[np.ix_(rows, native.snp_index)]
        assert native.total_samples == len(x)
        np.testing.assert_array_equal(native.to_numpy(), expected)
        _assert_statistics(native, expected)
        _assert_statistics(plink, expected)


@pytest.mark.parametrize("selection", ["prefix", "interior"])
@pytest.mark.parametrize("keep_file", [False, True])
def test_loader_qc_matches_independent_plink_command(
    tmp_path, synthetic_panel, selection, keep_file
):
    rows = _rows(synthetic_panel, selection)
    prefix, x = synthetic_panel["panels"]["missing"]
    keep = tmp_path / "samples.keep"
    keep.write_text("".join(f"F\t{s}\n" for s in synthetic_panel["samples"][rows]))
    output = tmp_path / "qc_reference"
    _run_plink(
        "plink2",
        "--bfile",
        prefix,
        "--keep",
        keep,
        "--maf",
        0.1,
        "--mac",
        3,
        "--write-snplist",
        "--out",
        output,
    )
    expected_snps = Path(str(output) + ".snplist").read_text().splitlines()
    af, n = _statistics(x[rows])
    keep_variants = (np.minimum(af, 1 - af) >= 0.1) & (
        2 * np.minimum(af, 1 - af) * n >= 3
    )
    assert expected_snps == list(synthetic_panel["snps"][keep_variants])
    for backend in ("magenpy", "plink"):
        kwargs = (
            {"keep_file": str(keep)}
            if keep_file
            else {"keep_samples": synthetic_panel["samples"][rows]}
        )
        loader = mgp.GWADataLoader(
            bed_files=prefix,
            backend=backend,
            min_maf=0.1,
            min_mac=3,
            temp_dir=str(tmp_path / backend),
            **kwargs,
        )
        assert (
            list(np.concatenate([g.snps for g in loader.genotype.values()]))
            == expected_snps
        )
        for g in loader.genotype.values():
            _assert_statistics(g, x[np.ix_(rows, g.snp_index)])


@pytest.mark.parametrize("kind", ["complete", "missing"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("weights", ["vector", "matrix", "canceling", "zero"])
def test_scores_after_splitting_match_plink_and_numpy(
    tmp_path, synthetic_panel, kind, standardize, weights
):
    # PLINK requires >=50 samples to estimate scoring frequencies implicitly.
    # Keep 56 while still excluding the source BED's last sample.
    rows = np.arange(4, 60)
    cols = np.array([8, 9, 10, 12, 14, 17, 19, 22])
    pair = _panel_pair(tmp_path, synthetic_panel, kind, rows, cols)
    plink, native = [g.split_by_variants({"region": g.snps})["region"] for g in pair]
    x = synthetic_panel["panels"][kind][1][np.ix_(rows, cols)]
    beta = np.linspace(-0.15, 0.2, len(cols))
    if weights == "matrix":
        beta = np.column_stack([beta, np.cos(np.arange(len(cols)) / 7.0)])
    elif weights == "canceling":
        beta = np.column_stack([beta, -beta])
    elif weights == "zero":
        beta = np.zeros((len(cols), 2))
    af, _ = _statistics(x)
    values = np.where(x >= 0, x, 2 * af)
    if standardize:
        values = (values - 2 * af) / np.sqrt(2 * af * (1 - af))
    expected = values @ beta
    observed = native.score(beta, standardize_genotype=standardize, impute_missing=True)
    reference = plink.score(beta, standardize_genotype=standardize)
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(observed, reference, atol=SCORE_ATOL, rtol=0)


def test_unimputed_score_matches_explicit_plink_option(tmp_path, synthetic_panel):
    rows = _rows(synthetic_panel, "strided")
    cols = np.array([8, 9, 12, 14, 17, 22])
    _, native = _panel_pair(tmp_path, synthetic_panel, "missing", rows, cols)
    prefix, x = synthetic_panel["panels"]["missing"]
    beta = np.linspace(-0.2, 0.3, len(cols))
    keep = tmp_path / "samples.keep"
    native.sample_table.get_individual_table().to_csv(
        keep, sep="\t", header=False, index=False
    )
    effects = tmp_path / "effects.tsv"
    pd.DataFrame({"SNP": native.snps, "A1": native.a1, "BETA": beta}).to_csv(
        effects, sep="\t", index=False
    )
    output = tmp_path / "score_reference"
    _run_plink(
        "plink2",
        "--bfile",
        prefix,
        "--keep",
        keep,
        "--score",
        effects,
        1,
        2,
        3,
        "header-read",
        "no-mean-imputation",
        "cols=+scoresums",
        "--out",
        output,
    )
    table = pd.read_csv(str(output) + ".sscore", sep=r"\s+").set_index("IID")
    reference = table.loc[native.samples].iloc[:, -1].to_numpy()
    np.testing.assert_allclose(native.score(beta), reference, atol=SCORE_ATOL, rtol=0)
    np.testing.assert_allclose(
        native.score(beta), np.maximum(x[np.ix_(rows, cols)], 0) @ beta
    )


@pytest.mark.parametrize("kind", ["complete", "missing"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize(
    "transform", ["none", "standardize_phenotype", "adjust_covariates", "outliers"]
)
def test_gwas_after_subsetting_and_splitting_matches_plink(
    tmp_path, synthetic_panel, kind, standardize, transform
):
    rows = _rows(synthetic_panel, "interior")
    cols = np.array([8, 9, 10, 12, 14, 17, 19, 22])
    pair = _panel_pair(tmp_path, synthetic_panel, kind, rows, cols)
    pair = [g.split_by_variants({"block": g.snps})["block"] for g in pair]
    x = synthetic_panel["panels"][kind][1][np.ix_(rows, cols)]
    t = np.arange(len(rows), dtype=float)
    y = 0.03 * t + np.sin(t / 3.0) + 0.2 * np.cos(t / 7.0)
    if transform == "outliers":
        y[5] = 50
    for g in pair:
        g.sample_table.set_phenotype(y, phenotype_likelihood="gaussian")
        g.sample_table.table["covariate"] = np.cos(t / 5.0)
        g.sample_table._covariate_cols = ["covariate"]
    kwargs = {} if transform == "none" else {transform: True}
    if transform == "outliers":
        kwargs = {"outlier_sigma_threshold": 2.5}
        selected = np.abs((y - y.mean()) / y.std()) <= 2.5
        assert not selected.all()
        x = x[selected]
    reference = pair[0].perform_gwas(**kwargs)
    observed = pair[1].perform_gwas(standardize_genotype=standardize, **kwargs)
    merged = _align_sumstats(reference, observed)
    np.testing.assert_array_equal(merged["N_magenpy"], merged["N_plink"])
    np.testing.assert_allclose(
        merged["MAF_magenpy"], merged["MAF_plink"], atol=OUTPUT_ATOL, rtol=0
    )
    for statistic in ("Z", "PVAL"):
        np.testing.assert_allclose(
            merged[f"{statistic}_magenpy"],
            merged[f"{statistic}_plink"],
            atol=OUTPUT_ATOL,
            rtol=0,
        )
    observed_x = np.where(x >= 0, x, np.nan)
    scale = np.nanstd(observed_x, axis=0) if standardize else np.ones(len(cols))
    for statistic in ("BETA", "SE"):
        np.testing.assert_allclose(
            merged[f"{statistic}_magenpy"],
            merged[f"{statistic}_plink"] * scale,
            atol=OUTPUT_ATOL,
            rtol=0,
        )


@pytest.mark.parametrize("kind", ["complete", "common"])
@pytest.mark.parametrize("estimator", ["sample", "windowed", "block"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "int16"])
@pytest.mark.parametrize("threads", [1, 2])
def test_ld_and_ld_scores_after_splitting_match_plink(
    tmp_path, synthetic_panel, kind, estimator, dtype, threads
):
    rows = _rows(synthetic_panel, "prefix")
    cols = np.arange(8, 24)
    pair = _panel_pair(tmp_path, synthetic_panel, kind, rows, cols, threads)
    plink, native = [g.split_by_variants({"region": g.snps})["region"] for g in pair]
    options = {"window_size": 8} if estimator == "windowed" else {}
    if estimator == "block":
        positions = native.bp_pos
        options = {
            "ld_blocks": np.array(
                [[positions[0], positions[8]], [positions[8], positions[-1] + 1]]
            )
        }
    reference = plink.compute_ld(
        estimator, str(tmp_path / "plink_ld"), dtype=dtype, **options
    )
    observed = native.compute_ld(
        estimator, str(tmp_path / "native_ld"), dtype=dtype, **options
    )
    assert reference.validate_ld_matrix() and observed.validate_ld_matrix()
    np.testing.assert_array_equal(observed.snps, reference.snps)
    assert observed.sample_size == reference.sample_size == len(rows)
    a = observed.to_csr(return_symmetric=True, dtype=np.float64).toarray()
    b = reference.to_csr(return_symmetric=True, dtype=np.float64).toarray()
    tolerance = (
        1.01 / np.iinfo(dtype).max
        if np.issubdtype(np.dtype(dtype), np.integer)
        else OUTPUT_ATOL
    )
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=0)
    # These relationships must survive every subset, stride, dtype, and backend.
    np.testing.assert_allclose(a[0, 1:3], [1, -1], atol=tolerance, rtol=0)
    for ld in (observed, reference):
        dense = ld.to_csr(return_symmetric=True, dtype=np.float64).toarray()
        np.testing.assert_allclose(
            ld.compute_ld_scores(corrected=False),
            np.square(dense).sum(axis=0),
            atol=2e-5,
            rtol=0,
        )
    for corrected in (False, True):
        np.testing.assert_allclose(
            observed.compute_ld_scores(corrected=corrected),
            reference.compute_ld_scores(corrected=corrected),
            atol=2 * len(cols) * tolerance,
            rtol=0,
        )


def test_blockwise_fallback_after_subsetting_matches_plink(tmp_path, synthetic_panel):
    from magenpy.stats.ld.estimator import BlockLD

    rows = _rows(synthetic_panel, "prefix")
    cols = np.arange(8, 24)
    plink, native = _panel_pair(tmp_path, synthetic_panel, "complete", rows, cols)
    positions = native.bp_pos
    blocks = np.array([[positions[0], positions[6]], [positions[9], positions[-1] + 1]])
    reference = plink.compute_ld(
        "block", str(tmp_path / "plink_blocks"), ld_blocks=blocks, dtype="float32"
    )
    observed = BlockLD(native, ld_blocks=blocks)._compute_blockwise(
        str(tmp_path / "native_blocks"),
        overwrite=True,
        delete_original=True,
        dtype="float32",
        compressor_name="zstd",
        compression_level=7,
        compute_spectral_properties=False,
        store_type=None,
    )
    np.testing.assert_array_equal(observed.snps, reference.snps)
    np.testing.assert_allclose(
        observed.to_csr(return_symmetric=True).toarray(),
        reference.to_csr(return_symmetric=True).toarray(),
        atol=OUTPUT_ATOL,
        rtol=0,
    )


@pytest.mark.parametrize("kind", ["complete", "common"])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_shrinkage_ld_after_subsetting_matches_plink(
    tmp_path, synthetic_panel, kind, dtype
):
    rows = _rows(synthetic_panel, "interior")
    cols = np.arange(8, 24)
    pair = _panel_pair(tmp_path, synthetic_panel, kind, rows, cols)
    results = []
    for g, name in zip(pair, ("plink", "native")):
        g.snp_table["cM"] = np.arange(g.n_snps, dtype=np.float32) * 0.001
        results.append(
            g.compute_ld(
                "shrinkage",
                str(tmp_path / name),
                dtype=dtype,
                genetic_map_ne=10000,
                genetic_map_sample_size=183,
            )
        )
    tolerance = 1.01 / 127 if dtype == "int8" else OUTPUT_ATOL
    np.testing.assert_allclose(
        results[1].to_csr(return_symmetric=True).toarray(),
        results[0].to_csr(return_symmetric=True).toarray(),
        atol=tolerance,
        rtol=0,
    )


@pytest.mark.parametrize("keep_original", [False, True])
def test_loader_sample_splits_match_plink_without_pooled_caches(
    tmp_path, synthetic_panel, keep_original
):
    prefix, x = synthetic_panel["panels"]["missing"]
    groups = [np.arange(24), np.arange(24, 48)]
    loaders = []
    for backend in ("plink", "magenpy"):
        loader = mgp.GWADataLoader(
            bed_files=prefix,
            backend=backend,
            min_maf=None,
            min_mac=None,
            temp_dir=str(tmp_path / backend),
        )
        for g in loader.genotype.values():
            _ = g.maf, g.n_per_snp
        splits = loader.split_by_samples(
            groups=[synthetic_panel["samples"][rows] for rows in groups],
            keep_original=keep_original,
        )
        assert loader.n == (synthetic_panel["n"] if keep_original else 24)
        for split, rows in zip(splits, groups):
            for g in split.genotype.values():
                _assert_statistics(g, x[np.ix_(rows, g.snp_index)])
        loaders.append(splits)
    for plink, native in zip(*loaders):
        for chromosome in native.chromosomes:
            np.testing.assert_allclose(
                native.genotype[chromosome].maf,
                plink.genotype[chromosome].maf,
                atol=OUTPUT_ATOL,
                rtol=0,
                equal_nan=True,
            )
    assert loaders[1][0].genotype[1].maf[0] == 0
    assert loaders[1][1].genotype[1].maf[0] == 1


@pytest.mark.parametrize("estimator", ["sample", "windowed", "block"])
def test_random_missing_ld_matches_independent_numpy(
    tmp_path, synthetic_panel, estimator
):
    # PLINK uses pairwise deletion; native LD uses mean-imputed, standardized
    # dosages. Random per-variant missingness therefore needs its own oracle.
    rows = _rows(synthetic_panel, "strided")
    cols = np.arange(8, 24)
    _, native = _panel_pair(tmp_path, synthetic_panel, "missing", rows, cols)
    native = native.split_by_variants({"block": native.snps})["block"]
    x = synthetic_panel["panels"]["missing"][1][np.ix_(rows, cols)]
    af, _ = _statistics(x)
    centered = np.where(x >= 0, x - 2 * af, 0)
    normalized = centered / np.sqrt(np.square(centered).sum(axis=0))
    expected = normalized.T @ normalized
    options = {}
    i, j = np.indices(expected.shape)
    if estimator == "windowed":
        options = {"window_size": 8}
        # Window upper bounds are exclusive; the stored triangle has offsets 1..7.
        expected[np.abs(i - j) >= 8] = 0
    elif estimator == "block":
        positions = native.bp_pos
        options = {
            "ld_blocks": np.array(
                [[positions[0], positions[8]], [positions[8], positions[-1] + 1]]
            )
        }
        expected[(i < 8) != (j < 8)] = 0
    ld = native.compute_ld(
        estimator, str(tmp_path / "native_ld"), dtype="float64", **options
    )
    np.testing.assert_allclose(
        ld.to_csr(return_symmetric=True, dtype=np.float64).toarray(),
        expected,
        atol=1e-10,
        rtol=0,
    )


def _numpy_linear_gwas(x, y, impute_missing, standardize):
    """Fit independent per-SNP OLS models with explicit intercepts."""
    from scipy.stats import t

    columns = ["N", "MAF", "BETA", "SE", "Z", "PVAL"]
    records = []
    for raw in x.T:
        observed = raw >= 0
        n_observed = int(observed.sum())
        row = dict.fromkeys(columns, np.nan)
        row["N"] = 0
        if n_observed:
            mean = raw[observed].mean()
            row["MAF"] = mean / 2
            values = np.where(observed, raw, mean) if impute_missing else raw[observed]
            phenotype = y if impute_missing else y[observed]
            row["N"] = len(values)
            if np.ptp(values) > 0:
                if standardize:
                    values = (values - values.mean()) / values.std()
                design = np.column_stack((np.ones(len(values)), values))
                coefficient = np.linalg.lstsq(design, phenotype, rcond=None)[0]
                row["BETA"] = coefficient[1]
                if len(values) > 2:
                    residual = phenotype - design @ coefficient
                    centered = values - values.mean()
                    row["SE"] = np.sqrt(
                        (residual @ residual)
                        / ((len(values) - 2) * (centered @ centered))
                    )
                    row["Z"] = row["BETA"] / row["SE"]
                    row["PVAL"] = 2 * t.sf(abs(row["Z"]), len(values) - 2)
        records.append(row)
    return pd.DataFrame(records, columns=columns)


@pytest.mark.parametrize("impute_missing", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("selection", ["prefix", "interior"])
def test_gwas_missing_call_modes_match_numpy_after_splits_and_outlier_removal(
    tmp_path, synthetic_panel, impute_missing, standardize, threads, selection
):
    prefix, x = synthetic_panel["panels"]["missing"]
    rows = _rows(synthetic_panel, selection)
    cols = np.array([8, 9, 10, 12, 14, 17, 19, 22])
    g = MagenpyGenotypeMatrix.from_file(prefix, temp_dir=str(tmp_path), threads=threads)
    _ = g.maf, g.n_per_snp  # Warm pooled caches before sample selection.
    g.filter_samples(keep_samples=synthetic_panel["samples"][rows])
    g = g.split_by_chromosome()[1]
    g = g.split_by_variants({"region": synthetic_panel["snps"][cols]})["region"]
    cached_maf, cached_n = g.maf.copy(), g.n_per_snp.copy()
    sample = np.arange(len(rows), dtype=float)
    y = 0.03 * sample + np.sin(sample / 3.0) + 0.2 * np.cos(sample / 7.0)
    y[3] = 50
    g.sample_table.set_phenotype(y, phenotype_likelihood="gaussian")
    retained = np.abs((y - y.mean()) / y.std()) <= 2.5
    assert not retained.all()
    expected = _numpy_linear_gwas(
        x[np.ix_(rows, cols)][retained], y[retained], impute_missing, standardize
    )
    observed = g.perform_gwas(
        impute_missing=impute_missing,
        standardize_genotype=standardize,
        outlier_sigma_threshold=2.5,
    ).to_table()
    np.testing.assert_array_equal(observed["SNP"], synthetic_panel["snps"][cols])
    np.testing.assert_array_equal(observed["N"], expected["N"])
    np.testing.assert_allclose(
        observed[expected.columns.drop("N")],
        expected.drop(columns="N"),
        rtol=1e-10,
        atol=1e-11,
        equal_nan=True,
    )
    # GWAS-only imputation/phenotype masks must not change genotype caches.
    np.testing.assert_array_equal(g.n_per_snp, cached_n)
    np.testing.assert_array_equal(g.maf, cached_maf)


@pytest.mark.parametrize("standardize", [False, True])
def test_gwas_missing_call_modes_agree_on_complete_snps(
    tmp_path, synthetic_panel, standardize
):
    prefix, x = synthetic_panel["panels"]["complete"]
    rows = _rows(synthetic_panel, "interior")
    cols = np.arange(8, 12)
    g = MagenpyGenotypeMatrix.from_file(prefix, temp_dir=str(tmp_path))
    g.filter_samples(keep_samples=synthetic_panel["samples"][rows])
    g.filter_snps(extract_snps=synthetic_panel["snps"][cols])
    assert np.all(x[np.ix_(rows, cols)] >= 0)
    y = np.sin(np.arange(len(rows)) / 3.0)
    g.sample_table.set_phenotype(y, phenotype_likelihood="gaussian")
    default = g.perform_gwas(standardize_genotype=standardize).to_table()
    for impute_missing in (False, True):
        observed = g.perform_gwas(
            standardize_genotype=standardize,
            impute_missing=impute_missing,
        ).to_table()
        pd.testing.assert_frame_equal(default, observed)


@pytest.mark.parametrize("impute_missing", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_gwas_imputation_handles_degenerate_snps(
    tmp_path, synthetic_panel, impute_missing, standardize
):
    prefix, x = synthetic_panel["panels"]["complete"]
    rows = _rows(synthetic_panel, "prefix")
    cols = np.array([2, 3, 5, 6, 7, 8, 12])
    g = MagenpyGenotypeMatrix.from_file(prefix, temp_dir=str(tmp_path))
    g.filter_samples(keep_samples=synthetic_panel["samples"][rows])
    g.filter_snps(extract_snps=synthetic_panel["snps"][cols])
    y = np.sin(np.arange(len(rows)) / 3.0)
    g.sample_table.set_phenotype(y, phenotype_likelihood="gaussian")
    expected = _numpy_linear_gwas(x[np.ix_(rows, cols)], y, impute_missing, standardize)
    observed = g.perform_gwas(
        impute_missing=impute_missing,
        standardize_genotype=standardize,
    ).to_table()
    np.testing.assert_array_equal(observed["N"], expected["N"])
    np.testing.assert_allclose(
        observed[expected.columns.drop("N")],
        expected.drop(columns="N"),
        rtol=1e-10,
        atol=1e-11,
        equal_nan=True,
    )
    assert observed.loc[0, "N"] == 0  # No observed mean exists for this SNP.
    assert np.isnan(observed.loc[0, "BETA"])
    assert np.isnan(observed.loc[3, "BETA"])  # One call cannot supply variation.
    assert np.isfinite(observed.loc[4, "SE"]) == impute_missing


@pytest.mark.parametrize("impute_missing", [False, True])
def test_loader_forwards_gwas_imputation_to_each_chromosome(
    tmp_path, synthetic_panel, impute_missing
):
    prefix, x = synthetic_panel["panels"]["missing"]
    rows = _rows(synthetic_panel, "prefix")
    cols = np.array([8, 12, 24, 31])
    loader = mgp.GWADataLoader(
        bed_files=prefix,
        backend="magenpy",
        min_maf=None,
        min_mac=None,
        keep_samples=synthetic_panel["samples"][rows],
        temp_dir=str(tmp_path),
    )
    loader.filter_snps(extract_snps=synthetic_panel["snps"][cols])
    y = np.sin(np.arange(len(rows)) / 3.0)
    loader.set_phenotype(y, phenotype_likelihood="gaussian")
    loader.perform_gwas(impute_missing=impute_missing)
    for chromosome, g in loader.genotype.items():
        expected = _numpy_linear_gwas(
            x[np.ix_(rows, g.snp_index)], y, impute_missing, False
        )
        observed = loader.sumstats_table[chromosome].to_table()
        np.testing.assert_array_equal(observed["N"], expected["N"])
        np.testing.assert_allclose(
            observed[expected.columns.drop("N")],
            expected.drop(columns="N"),
            rtol=1e-10,
            atol=1e-11,
        )


@pytest.mark.parametrize("impute_missing", [False, True])
@pytest.mark.parametrize("empty", [None, "samples", "snps"])
def test_gwas_kernel_optional_outputs_and_empty_inputs(
    tmp_path, synthetic_panel, impute_missing, empty
):
    from magenpy.stats.variant.variant_cpp import compute_gwa_linear_stats

    prefix, _ = synthetic_panel["panels"]["missing"]
    rows = np.arange(0 if empty == "samples" else 24, dtype=np.int32)
    cols = np.array([] if empty == "snps" else [2, 3, 6, 7, 8, 12], dtype=np.int32)
    y = np.sin(np.arange(len(rows)) / 3.0)
    if len(y):
        y -= y.mean()
    args = (prefix + ".bed", cols, rows, synthetic_panel["n"], y)
    options = dict(impute_missing=impute_missing, standardize_genotype=True)
    original = compute_gwa_linear_stats(*args, **options)
    extended = compute_gwa_linear_stats(*args, return_y_sum_sq=True, **options)
    assert len(original) == 4 and len(extended) == 5
    for a, b in zip(original, extended):
        np.testing.assert_array_equal(a, b)
        assert a.shape == (len(cols),)
    if empty == "samples":
        np.testing.assert_array_equal(extended[1], np.zeros(len(cols), dtype=np.int32))
        assert np.isnan(extended[4]).all()
    if empty == "snps":
        assert extended[4].shape == (0,)
