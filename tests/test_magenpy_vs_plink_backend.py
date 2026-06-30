import shutil

import numpy as np
import pandas as pd
import pytest

import magenpy as mgp
from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix, plinkBEDGenotypeMatrix


pytestmark = pytest.mark.plink

OUTPUT_ATOL = 1e-5
SCORE_ATOL = 1e-4


def _require_native_extensions():
    variant_cpp = pytest.importorskip("magenpy.stats.variant.variant_cpp")
    score_cpp = pytest.importorskip("magenpy.stats.score.score_cpp")
    ld_cpp = pytest.importorskip("magenpy.stats.ld.c_utils")

    if not hasattr(variant_cpp, "extract_genotype_matrix"):
        pytest.skip("magenpy.stats.variant.variant_cpp must be rebuilt from Cython.")
    if not hasattr(variant_cpp, "compute_gwa_linear_stats"):
        pytest.skip("magenpy.stats.variant.variant_cpp must expose compute_gwa_linear_stats.")
    if not hasattr(score_cpp, "calculate_pgs"):
        pytest.skip("magenpy.stats.score.score_cpp is unavailable.")
    if not hasattr(ld_cpp, "compute_ut_ld_from_bed"):
        pytest.skip("magenpy.stats.ld.c_utils must be rebuilt from Cython.")


def _require_plink():
    if shutil.which("plink2") is None:
        pytest.skip("plink2 executable is not available on PATH.")
    if shutil.which("plink") is None:
        pytest.skip("plink1.9 executable is not available on PATH.")


@pytest.fixture(scope="module")
def backend_comparison_inputs(tmp_path_factory):
    _require_native_extensions()
    _require_plink()

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
        pytest.skip("Reference dataset subset does not contain enough complete-call variants.")

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
            0.03 * sample_idx
            + np.sin(sample_idx / 5.0)
            + np.cos(sample_idx / 11.0)
        )
        gmat.sample_table.set_phenotype(phenotype, phenotype_likelihood="gaussian")

    return plink_gmat, magenpy_gmat


def _align_sumstats(plink_sumstats, magenpy_sumstats):
    plink_table = plink_sumstats.to_table(["SNP", "MAF", "N", "BETA", "SE"])
    magenpy_table = magenpy_sumstats.to_table(["SNP", "MAF", "N", "BETA", "SE"])

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
def test_score_matches_plink2(tmp_path, backend_comparison_inputs, standardize_genotype):
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


def test_windowed_ld_matches_plink1p9_on_complete_variants(tmp_path, backend_comparison_inputs):
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
