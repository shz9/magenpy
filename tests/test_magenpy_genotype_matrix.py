import shutil

import numpy as np
import pytest

import magenpy as mgp
from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix
from magenpy.stats.ld.estimator import WindowedLD


def _require_native_extensions():
    variant_cpp = pytest.importorskip("magenpy.stats.variant.variant_cpp")
    score_cpp = pytest.importorskip("magenpy.stats.score.score_cpp")
    ld_cpp = pytest.importorskip("magenpy.stats.ld.c_utils")

    if not hasattr(variant_cpp, "extract_genotype_matrix"):
        pytest.skip("magenpy.stats.variant.variant_cpp must be rebuilt from Cython.")
    if not hasattr(score_cpp, "calculate_pgs"):
        pytest.skip("magenpy.stats.score.score_cpp is unavailable.")
    if not hasattr(ld_cpp, "compute_ut_ld_from_bed"):
        pytest.skip("magenpy.stats.ld.c_utils must be rebuilt from Cython.")


@pytest.fixture(scope="module")
def magenpy_gmat(tmp_path_factory):
    _require_native_extensions()

    temp_dir = tmp_path_factory.mktemp("magenpy_gmat_tmp")
    gmat = MagenpyGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(temp_dir),
        threads=2,
    )

    gmat.filter_samples(keep_samples=gmat.samples[:48])
    gmat.filter_snps(extract_snps=gmat.snps[:64])

    yield gmat

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def genotype_numpy(magenpy_gmat):
    return magenpy_gmat.to_numpy(dtype=np.int8)


def _allele_frequency_and_n(genotypes):
    observed = genotypes >= 0
    n_per_snp = observed.sum(axis=0).astype(np.int32)
    dosage_sum = np.where(observed, genotypes, 0).sum(axis=0)
    allele_frequency = np.divide(
        dosage_sum,
        2.0 * n_per_snp,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=n_per_snp > 0,
    )
    return allele_frequency, n_per_snp


def _imputed_standardized_genotypes(genotypes):
    allele_frequency, _ = _allele_frequency_and_n(genotypes)
    mean = 2.0 * allele_frequency
    imputed = np.where(genotypes >= 0, genotypes, mean)
    centered = imputed - mean
    sum_squares = np.square(centered).sum(axis=0)
    inverse_sd = np.divide(
        np.sqrt(genotypes.shape[0]),
        np.sqrt(sum_squares),
        out=np.zeros(genotypes.shape[1], dtype=np.float64),
        where=sum_squares > 0,
    )
    return centered * inverse_sd


def test_to_numpy_respects_dtype(magenpy_gmat, genotype_numpy):
    assert genotype_numpy.shape == magenpy_gmat.shape
    assert genotype_numpy.dtype == np.dtype(np.int8)
    assert np.all(np.isin(genotype_numpy, [-1, 0, 1, 2]))

    as_int16 = magenpy_gmat.to_numpy(dtype=np.int16)
    assert as_int16.dtype == np.dtype(np.int16)
    assert np.array_equal(as_int16, genotype_numpy.astype(np.int16))

    as_float32 = magenpy_gmat.to_numpy(dtype=np.float32)
    assert as_float32.dtype == np.dtype(np.float32)
    assert np.array_equal(np.isnan(as_float32), genotype_numpy < 0)
    assert np.array_equal(as_float32[genotype_numpy >= 0],
                          genotype_numpy[genotype_numpy >= 0].astype(np.float32))


def test_compute_variant_statistics(magenpy_gmat, genotype_numpy):
    expected_maf, expected_n = _allele_frequency_and_n(genotype_numpy)

    magenpy_gmat.compute_allele_frequency()
    magenpy_gmat.compute_sample_size_per_snp()

    np.testing.assert_allclose(magenpy_gmat.maf, expected_maf, equal_nan=True)
    assert np.array_equal(magenpy_gmat.n_per_snp, expected_n)


def test_score_matches_numpy(genotype_numpy, magenpy_gmat):
    beta = np.column_stack([
        np.linspace(-0.2, 0.3, magenpy_gmat.n_snps),
        np.cos(np.arange(magenpy_gmat.n_snps) / 7.0),
    ])

    raw_genotypes = np.where(genotype_numpy > 0, genotype_numpy, 0).astype(np.float64)
    expected_raw = raw_genotypes @ beta
    np.testing.assert_allclose(magenpy_gmat.score(beta), expected_raw, rtol=1e-10, atol=1e-10)

    allele_frequency, _ = _allele_frequency_and_n(genotype_numpy)
    mean = 2.0 * allele_frequency
    imputed_genotypes = np.where(genotype_numpy >= 0, genotype_numpy, mean)
    expected_imputed = imputed_genotypes @ beta
    np.testing.assert_allclose(
        magenpy_gmat.score(beta, impute_missing=True),
        expected_imputed,
        rtol=1e-10,
        atol=1e-10,
    )

    variance = mean * (1.0 - allele_frequency)
    inverse_sd = np.divide(
        1.0,
        np.sqrt(variance),
        out=np.zeros_like(variance, dtype=np.float64),
        where=variance > 0,
    )
    standardized = np.where(
        genotype_numpy >= 0,
        (genotype_numpy - mean) * inverse_sd,
        0.0,
    )
    expected_standardized = standardized @ beta
    np.testing.assert_allclose(
        magenpy_gmat.score(beta, standardize_genotype=True),
        expected_standardized,
        rtol=1e-10,
        atol=1e-10,
    )


def test_compute_ld_matches_numpy_windowed(tmp_path, genotype_numpy, magenpy_gmat):
    window_size = 8
    ld_boundaries = WindowedLD(magenpy_gmat, window_size=window_size).compute_ld_boundaries()

    expected_full = _imputed_standardized_genotypes(genotype_numpy)
    expected_full = expected_full.T @ expected_full / genotype_numpy.shape[0]

    expected_sparse = np.zeros_like(expected_full)
    np.fill_diagonal(expected_sparse, 1.0)
    for row in range(magenpy_gmat.n_snps):
        row_end = int(ld_boundaries[1, row])
        if row_end <= row + 1:
            continue
        expected_sparse[row, row + 1:row_end] = expected_full[row, row + 1:row_end]
        expected_sparse[row + 1:row_end, row] = expected_full[row + 1:row_end, row]

    ld_mat = magenpy_gmat.compute_ld(
        "windowed",
        str(tmp_path / "native_ld"),
        dtype="float32",
        window_size=window_size,
    )

    assert ld_mat.validate_ld_matrix()
    assert ld_mat.stored_n_snps == magenpy_gmat.n_snps
    assert ld_mat.sample_size == magenpy_gmat.sample_size
    assert np.array_equal(ld_mat.snps, magenpy_gmat.snps)

    observed = ld_mat.to_csr(return_symmetric=True, dtype=np.float32).toarray()
    np.testing.assert_allclose(observed, expected_sparse, rtol=2e-5, atol=2e-5)
