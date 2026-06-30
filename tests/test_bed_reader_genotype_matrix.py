import shutil

import numpy as np
import pytest

import magenpy as mgp
from magenpy.GenotypeMatrix import bedReaderGenotypeMatrix
from magenpy.stats.ld.estimator import WindowedLD


def _require_bed_reader():
    pytest.importorskip("bed_reader")


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


def _ld_standardized_genotypes(genotypes):
    allele_frequency, _ = _allele_frequency_and_n(genotypes)
    mean = 2.0 * allele_frequency
    centered = np.where(genotypes >= 0, genotypes - mean, 0.0)
    sum_squares = np.square(centered).sum(axis=0)
    inverse_sd = np.divide(
        np.sqrt(genotypes.shape[0]),
        np.sqrt(sum_squares),
        out=np.zeros(genotypes.shape[1], dtype=np.float64),
        where=sum_squares > 0,
    )
    return centered * inverse_sd


def _expected_gwas(genotypes, phenotype, standardize_genotype=False):
    observed = genotypes >= 0
    n_per_snp = observed.sum(axis=0).astype(np.int32)
    dosage_sum = np.where(observed, genotypes, 0.0).sum(axis=0)
    mean = np.divide(
        dosage_sum,
        n_per_snp,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=n_per_snp > 0,
    )
    x = np.where(observed, genotypes - mean, 0.0)

    if standardize_genotype:
        sd = np.sqrt(np.divide(
            np.square(x).sum(axis=0),
            n_per_snp,
            out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
            where=n_per_snp > 0,
        ))
        x = np.divide(
            x,
            sd,
            out=np.zeros_like(x, dtype=np.float64),
            where=sd > 0,
        )

    centered_phenotype = phenotype - phenotype.mean()
    x_dot_y = x.T.dot(centered_phenotype)
    sum_x_sq = np.square(x).sum(axis=0)
    slope = np.divide(
        x_dot_y,
        sum_x_sq,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=sum_x_sq > 0,
    )

    residual_sum_sq = np.dot(centered_phenotype, centered_phenotype) - slope * x_dot_y
    residual_sum_sq = np.maximum(residual_sum_sq, 0.0)
    s2 = np.divide(
        residual_sum_sq,
        n_per_snp - 2,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=n_per_snp > 2,
    )
    se = np.sqrt(np.divide(
        s2,
        sum_x_sq,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=sum_x_sq > 0,
    ))
    maf = np.divide(
        dosage_sum,
        2.0 * n_per_snp,
        out=np.full(genotypes.shape[1], np.nan, dtype=np.float64),
        where=n_per_snp > 0,
    )

    return maf, n_per_snp, slope, se


@pytest.fixture(scope="module")
def bed_reader_gmat(tmp_path_factory):
    _require_bed_reader()

    temp_dir = tmp_path_factory.mktemp("bed_reader_gmat_tmp")
    gmat = bedReaderGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(temp_dir),
        threads=2,
    )

    gmat.filter_samples(keep_samples=gmat.samples[:48])
    gmat.filter_snps(extract_snps=gmat.snps[:64])

    yield gmat

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def genotype_numpy(bed_reader_gmat):
    return bed_reader_gmat.to_numpy(dtype=np.int8)


def test_to_numpy_and_csr_respect_dtype(bed_reader_gmat, genotype_numpy):
    assert genotype_numpy.shape == bed_reader_gmat.shape
    assert genotype_numpy.dtype == np.dtype(np.int8)
    assert np.all(np.isin(genotype_numpy, [-1, 0, 1, 2]))

    as_float32 = bed_reader_gmat.to_numpy(dtype=np.float32)
    assert as_float32.dtype == np.dtype(np.float32)
    assert np.array_equal(np.isnan(as_float32), genotype_numpy < 0)
    assert np.array_equal(
        as_float32[genotype_numpy >= 0],
        genotype_numpy[genotype_numpy >= 0].astype(np.float32),
    )

    csr = bed_reader_gmat.to_csr(dtype=np.int8)
    assert csr.getformat() == "csr"
    assert csr.shape == bed_reader_gmat.shape
    assert csr.dtype == np.dtype(np.int8)
    assert -127 not in csr.data
    np.testing.assert_array_equal(csr.toarray(), genotype_numpy)


def test_compute_variant_statistics(bed_reader_gmat, genotype_numpy):
    expected_maf, expected_n = _allele_frequency_and_n(genotype_numpy)

    bed_reader_gmat.compute_allele_frequency()
    bed_reader_gmat.compute_sample_size_per_snp()

    np.testing.assert_allclose(bed_reader_gmat.maf, expected_maf, equal_nan=True)
    np.testing.assert_array_equal(bed_reader_gmat.n_per_snp, expected_n)


def test_score_matches_numpy(bed_reader_gmat, genotype_numpy):
    beta = np.column_stack([
        np.linspace(-0.2, 0.3, bed_reader_gmat.n_snps),
        np.cos(np.arange(bed_reader_gmat.n_snps) / 7.0),
    ])

    raw_genotypes = np.where(genotype_numpy > 0, genotype_numpy, 0).astype(np.float64)
    expected_raw = raw_genotypes @ beta

    np.testing.assert_allclose(
        bed_reader_gmat.score(beta),
        expected_raw,
        rtol=1e-10,
        atol=1e-10,
    )


def test_standardized_score_matches_numpy_for_complete_variants(tmp_path):
    _require_bed_reader()

    gmat = bedReaderGenotypeMatrix.from_file(
        mgp.tgp_eur_data_path(),
        temp_dir=str(tmp_path),
        threads=2,
    )
    gmat.filter_samples(keep_samples=gmat.samples[:48])

    discovery = gmat.to_numpy(dtype=np.int8)
    complete_snps = gmat.snps[np.all(discovery >= 0, axis=0)][:32]
    if complete_snps.shape[0] < 8:
        pytest.skip("Reference data subset does not contain enough complete variants.")

    gmat.filter_snps(extract_snps=complete_snps)
    genotypes = gmat.to_numpy(dtype=np.float64)
    beta = np.linspace(-0.2, 0.3, gmat.n_snps)
    expected = ((genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)).dot(beta)

    np.testing.assert_allclose(
        gmat.score(beta, standardize_genotype=True),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("standardize_genotype", [False, True])
def test_perform_gwas_matches_numpy(bed_reader_gmat, genotype_numpy, standardize_genotype):
    sample_idx = np.arange(bed_reader_gmat.sample_size, dtype=np.float64)
    phenotype = 0.03 * sample_idx + np.sin(sample_idx / 5.0)
    bed_reader_gmat.sample_table.set_phenotype(
        phenotype,
        phenotype_likelihood="gaussian",
    )

    sumstats = bed_reader_gmat.perform_gwas(
        standardize_genotype=standardize_genotype,
    )
    table = sumstats.to_table(["SNP", "MAF", "N", "BETA", "SE"])
    expected_maf, expected_n, expected_beta, expected_se = _expected_gwas(
        genotype_numpy,
        phenotype,
        standardize_genotype=standardize_genotype,
    )

    np.testing.assert_array_equal(table["SNP"].values, bed_reader_gmat.snps)
    np.testing.assert_allclose(table["MAF"].values, expected_maf, equal_nan=True)
    np.testing.assert_array_equal(table["N"].values, expected_n)
    np.testing.assert_allclose(table["BETA"].values, expected_beta, equal_nan=True)
    np.testing.assert_allclose(table["SE"].values, expected_se, equal_nan=True)


def test_compute_ld_matches_numpy_windowed(tmp_path, bed_reader_gmat, genotype_numpy):
    window_size = 8
    ld_boundaries = WindowedLD(
        bed_reader_gmat,
        window_size=window_size,
    ).compute_ld_boundaries()

    expected_full = _ld_standardized_genotypes(genotype_numpy)
    expected_full = expected_full.T @ expected_full / genotype_numpy.shape[0]

    expected_sparse = np.zeros_like(expected_full)
    np.fill_diagonal(expected_sparse, 1.0)
    for row in range(bed_reader_gmat.n_snps):
        row_end = int(ld_boundaries[1, row])
        if row_end <= row + 1:
            continue
        expected_sparse[row, row + 1:row_end] = expected_full[row, row + 1:row_end]
        expected_sparse[row + 1:row_end, row] = expected_full[row + 1:row_end, row]

    ld_mat = bed_reader_gmat.compute_ld(
        "windowed",
        str(tmp_path / "bed_reader_ld"),
        dtype="float32",
        window_size=window_size,
    )

    assert ld_mat.validate_ld_matrix()
    assert ld_mat.stored_n_snps == bed_reader_gmat.n_snps
    assert ld_mat.sample_size == bed_reader_gmat.sample_size
    np.testing.assert_array_equal(ld_mat.snps, bed_reader_gmat.snps)

    observed = ld_mat.to_csr(return_symmetric=True, dtype=np.float32).toarray()
    np.testing.assert_allclose(observed, expected_sparse, rtol=2e-5, atol=2e-5)
