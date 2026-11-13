"""Filtering regressions using independently encoded, known PLINK dosages."""

import copy

import numpy as np
import pytest

import magenpy as mgp
from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix


@pytest.fixture
def known_bed(tmp_path):
    # 13 source samples require four bytes/SNP; neither ancestry retains the
    # last sample. Inferring either subset's stride silently reads wrong bytes.
    rng = np.random.default_rng(71)
    x = rng.integers(0, 3, size=(13, 10), dtype=np.int8)
    x[:6, 0] = 0
    x[6:, 0] = 2
    x[:6, 1] = 2
    x[6:, 1] = 0
    x[:, 2] = -1
    x[1, 4] = -1
    x[8, 7] = -1
    prefix = tmp_path / "known"
    prefix.with_suffix(".fam").write_text(
        "".join(f"F s{i} 0 0 1 -9\n" for i in range(len(x)))
    )
    prefix.with_suffix(".bim").write_text(
        "".join(
            f"{1 + (j >= 5)}\trs{j}\t0\t{100 + j * 10}\tA\tG\n"
            for j in range(x.shape[1])
        )
    )
    data = bytearray(b"\x6c\x1b\x01")
    codes = {-1: 1, 0: 3, 1: 2, 2: 0}
    for column in x.T:
        for start in range(0, len(x), 4):
            data.append(
                sum(
                    codes[int(v)] << (2 * k)
                    for k, v in enumerate(column[start : start + 4])
                )
            )
    prefix.with_suffix(".bed").write_bytes(data)
    return str(prefix), x


def stats(x):
    n = (x >= 0).sum(axis=0)
    af = np.divide(
        np.maximum(x, 0).sum(axis=0),
        2 * n,
        out=np.full(x.shape[1], np.nan),
        where=n > 0,
    )
    return af, n


def matrix(known_bed, tmp_path):
    return MagenpyGenotypeMatrix.from_file(known_bed[0], temp_dir=str(tmp_path))


@pytest.mark.parametrize("use_file", [False, True])
def test_loader_ancestry_qc(known_bed, tmp_path, use_file):
    prefix, x = known_bed
    for rows in (np.arange(6), np.arange(6, 12)):
        ids = np.array([f"s{i}" for i in rows])
        if use_file:
            keep = tmp_path / "keep.txt"
            keep.write_text("".join(f"F\t{i}\n" for i in ids))
            kwargs = {"keep_file": str(keep)}
        else:
            kwargs = {"keep_samples": ids}
        loader = mgp.GWADataLoader(
            bed_files=prefix, min_maf=0.1, min_mac=2, temp_dir=str(tmp_path), **kwargs
        )
        af, n = stats(x[rows])
        mac = np.minimum(af, 1 - af) * 2 * n
        valid = np.isfinite(af) & (np.minimum(af, 1 - af) >= 0.1) & (mac >= 2)
        for child in loader.genotype.values():
            cols = child.snp_index
            expected_cols = np.flatnonzero(
                valid & ((np.arange(10) >= 5) == (child.chromosome == 2))
            )
            np.testing.assert_array_equal(cols, expected_cols)
            np.testing.assert_allclose(child.maf, af[cols])
            np.testing.assert_array_equal(child.n_per_snp, n[cols])
            np.testing.assert_array_equal(child.to_numpy(), x[np.ix_(rows, cols)])


def test_cache_invalidation_and_split_isolation(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    x = known_bed[1]
    _ = g.maf, g.n_per_snp
    children = g.split_by_variants({"a": ["rs4", "rs0", "rs4", "rs8"], "b": ["rs1"]})
    a = children["a"]
    np.testing.assert_array_equal(a.snp_index, [0, 4, 8])
    a.filter_samples(keep_samples=["s5", "s1", "s1", "absent"])
    np.testing.assert_array_equal(a.sample_index, [1, 5])
    assert g.n == children["b"].n == 13
    assert "MAF" not in a.snp_table and "N" not in a.snp_table
    af, n = stats(x[np.ix_([1, 5], [0, 4, 8])])
    np.testing.assert_allclose(a.maf, af)
    np.testing.assert_array_equal(a.n_per_snp, n)
    assert a.total_samples == 13
    # Direct constructor with subset metadata must also use the source FAM.
    rebuilt = MagenpyGenotypeMatrix(
        sample_table=a.sample_table,
        snp_table=a.snp_table,
        bed_file=a.bed_file,
        temp_dir=str(tmp_path),
    )
    assert rebuilt.total_samples == 13
    np.testing.assert_array_equal(rebuilt.to_numpy(), x[np.ix_([1, 5], [0, 4, 8])])
    # Repeated/no-op selection preserves valid caches, including metadata updates.
    a.filter_samples(keep_samples=a.samples)
    assert "MAF" in a.snp_table
    replacement = copy.deepcopy(g.sample_table)
    replacement.table = replacement.table.iloc[[2, 4, 7]].copy()
    g.set_sample_table(replacement)
    np.testing.assert_allclose(g.maf, stats(x[[2, 4, 7]])[0], equal_nan=True)


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_filtered_split_ld_scores_and_scoring(known_bed, tmp_path, dtype):
    g = matrix(known_bed, tmp_path)
    _ = g.maf, g.n_per_snp
    g.filter_samples(keep_samples=[f"s{i}" for i in range(6)])
    child = g.split_by_variants({"block": ["rs9", "rs3", "rs7", "rs4"]})["block"]
    # A block is conventionally confined to one chromosome.
    child.snp_table["CHR"] = 1
    x = known_bed[1][:6, child.snp_index]
    af, n = stats(x)
    np.testing.assert_allclose(child.maf, af)
    np.testing.assert_array_equal(child.n_per_snp, n)
    beta = np.arange(1.0, 5.0)
    np.testing.assert_allclose(child.score(beta), np.maximum(x, 0) @ beta)
    imputed = np.where(x >= 0, x, 2 * af)
    np.testing.assert_allclose(child.score(beta, impute_missing=True), imputed @ beta)
    z = (imputed - 2 * af) / np.sqrt(2 * af * (1 - af))
    np.testing.assert_allclose(child.score(beta, standardize_genotype=True), z @ beta)
    centered = imputed - 2 * af
    normalized = centered / np.sqrt(np.square(centered).sum(axis=0))
    expected = normalized.T @ normalized
    ld = child.compute_ld(
        "windowed", str(tmp_path / dtype), dtype=dtype, window_size=20
    )
    actual = ld.to_csr(return_symmetric=True, dtype=np.float32).toarray()
    tolerance = 1 / 127 if dtype == "int8" else 2e-6
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(
        ld.compute_ld_scores(corrected=False),
        np.square(expected).sum(axis=0),
        atol=0.05 if dtype == "int8" else 2e-6,
    )


def test_loader_sync_after_inplace_sample_change(known_bed, tmp_path):
    loader = mgp.GWADataLoader(bed_files=known_bed[0], temp_dir=str(tmp_path))
    for g in loader.genotype.values():
        _ = g.maf, g.n_per_snp
    loader.sample_table.filter_samples(keep_samples=["s1", "s3", "s8"])
    loader.sync_sample_tables()
    for g in loader.genotype.values():
        expected = known_bed[1][np.ix_([1, 3, 8], g.snp_index)]
        af, n = stats(expected)
        np.testing.assert_allclose(g.maf, af, equal_nan=True)
        np.testing.assert_array_equal(g.n_per_snp, n)


def test_empty_and_duplicate_filters(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    g.filter_snps(extract_snps=["rs7", "rs3", "rs7", "unknown"])
    np.testing.assert_array_equal(g.snp_index, [3, 7])
    g.filter_samples(keep_samples=[])
    assert g.to_numpy().shape == (0, 2)
    assert np.all(np.isnan(g.maf))
    np.testing.assert_array_equal(g.n_per_snp, [0, 0])
    g.filter_by_allele_frequency(min_mac=1)
    assert g.n_snps == 0
    assert g.to_numpy().shape == (0, 0)


@pytest.mark.parametrize("indices", [[-1, 2], [1, 13], [2, 1], [1, 1]])
def test_invalid_native_sample_indices(known_bed, tmp_path, indices):
    g = matrix(known_bed, tmp_path)
    g.sample_table.table = g.sample_table.table.iloc[:2].copy()
    g.sample_table.table["original_index"] = indices
    with pytest.raises(ValueError):
        g.to_numpy()


@pytest.mark.parametrize(
    "kwargs",
    [{"min_maf": -0.1}, {"min_maf": 0.6}, {"min_mac": -1}, {"min_mac": np.nan}],
)
def test_invalid_qc_thresholds(known_bed, tmp_path, kwargs):
    with pytest.raises(ValueError):
        matrix(known_bed, tmp_path).filter_by_allele_frequency(**kwargs)


def test_phenotype_selection_precedes_qc(known_bed, tmp_path):
    phenotype = tmp_path / "phenotype.txt"
    phenotype.write_text(
        "FID\tIID\tphenotype\n" + "".join(f"F\ts{i}\t{i}\n" for i in range(6))
    )
    loader = mgp.GWADataLoader(
        bed_files=known_bed[0],
        phenotype_file=str(phenotype),
        min_mac=1,
        temp_dir=str(tmp_path),
    )
    af, n = stats(known_bed[1][:6])
    valid = np.isfinite(af) & (np.minimum(af, 1 - af) * 2 * n >= 1)
    assert loader.n == 6
    np.testing.assert_array_equal(
        np.concatenate([g.snp_index for g in loader.genotype.values()]),
        np.flatnonzero(valid),
    )


@pytest.mark.parametrize("backend", ["bed-reader", "xarray"])
def test_optional_backend_filters_and_splits(known_bed, tmp_path, backend):
    if backend == "bed-reader":
        pytest.importorskip("bed_reader")
        from magenpy.GenotypeMatrix import bedReaderGenotypeMatrix

        g = bedReaderGenotypeMatrix.from_file(known_bed[0], temp_dir=str(tmp_path))
    else:
        xr = pytest.importorskip("xarray")
        da = pytest.importorskip("dask.array")
        from magenpy.GenotypeMatrix import xarrayGenotypeMatrix

        template = matrix(known_bed, tmp_path)
        x = known_bed[1].astype(float)
        x[x < 0] = np.nan
        array = xr.DataArray(
            da.from_array(x, chunks=(4, 4)),
            dims=("sample", "variant"),
            coords={"sample": template.samples, "snp": ("variant", template.snps)},
        )
        g = xarrayGenotypeMatrix(
            sample_table=template.sample_table,
            snp_table=template.snp_table,
            xr_mat=array,
            temp_dir=str(tmp_path),
        )
    _ = g.maf, g.n_per_snp
    g.filter_samples(keep_samples=[f"s{i}" for i in range(6)])
    af, n = stats(known_bed[1][:6])
    np.testing.assert_allclose(g.maf, af, equal_nan=True)
    np.testing.assert_array_equal(g.n_per_snp, n)
    g.filter_by_allele_frequency(min_mac=1)
    cols = np.flatnonzero(np.isfinite(af) & (np.minimum(af, 1 - af) * 2 * n >= 1))
    np.testing.assert_array_equal(g.snp_index, cols)
    expected = known_bed[1][:6, cols].astype(float)
    expected[expected < 0] = np.nan
    np.testing.assert_allclose(g.to_numpy(dtype=float), expected, equal_nan=True)
    for child in g.split_by_chromosome().values():
        child_expected = known_bed[1][:6, child.snp_index].astype(float)
        child_expected[child_expected < 0] = np.nan
        np.testing.assert_allclose(
            child.to_numpy(dtype=float), child_expected, equal_nan=True
        )
    child = g.split_by_variants({"a": g.snps[::2]})["a"]
    np.testing.assert_allclose(
        child.to_numpy(dtype=float), expected[:, ::2], equal_nan=True
    )
    # SNP extraction must slice the same positions as metadata.
    g.filter_snps(extract_snps=g.snps[::2])
    np.testing.assert_allclose(
        g.to_numpy(dtype=float), expected[:, ::2], equal_nan=True
    )


def test_ancestry_frequencies_and_loader_sample_splits(known_bed, tmp_path):
    loader = mgp.GWADataLoader(bed_files=known_bed[0], temp_dir=str(tmp_path))
    for g in loader.genotype.values():
        _ = g.maf, g.n_per_snp
    groups = [[f"s{i}" for i in range(6)], [f"s{i}" for i in range(6, 12)]]
    splits = loader.split_by_samples(groups=groups)
    assert loader.n == 13
    for split, rows in zip(splits, [np.arange(6), np.arange(6, 12)]):
        for g in split.genotype.values():
            af, n = stats(known_bed[1][np.ix_(rows, g.snp_index)])
            np.testing.assert_allclose(g.maf, af, equal_nan=True)
            np.testing.assert_array_equal(g.n_per_snp, n)
    assert splits[0].genotype[1].maf[0] == 0
    assert splits[1].genotype[1].maf[0] == 1


def test_gwas_after_sample_and_variant_splitting(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    _ = g.maf, g.n_per_snp
    g.filter_samples(keep_samples=[f"s{i}" for i in range(6)])
    child = g.split_by_variants({"a": ["rs3", "rs4", "rs7", "rs9"]})["a"]
    phenotype = np.array([0.2, 0.9, -0.3, 1.4, 0.6, -0.2])
    child.sample_table.set_phenotype(phenotype, phenotype_likelihood="gaussian")
    observed = child.perform_gwas().to_table()
    x = known_bed[1][:6, child.snp_index]
    af, n = stats(x)
    centered = np.where(x >= 0, x - 2 * af, 0)
    sums = np.square(centered).sum(axis=0)
    beta = centered.T @ (phenotype - phenotype.mean()) / sums
    observed_y = np.where(x >= 0, phenotype[:, None], np.nan)
    centered_y = observed_y - np.nanmean(observed_y, axis=0)
    residual = np.nansum(np.square(centered_y), axis=0) - np.square(beta) * sums
    se = np.sqrt(residual / (n - 2) / sums)
    np.testing.assert_allclose(observed["MAF"], af)
    np.testing.assert_array_equal(observed["N"], n)
    np.testing.assert_allclose(observed["BETA"], beta, atol=1e-12)
    np.testing.assert_allclose(observed["SE"], se)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_frequency_resolution_reuses_readonly_input(
    known_bed, tmp_path, dtype, strided
):
    g = matrix(known_bed, tmp_path)
    frequencies = np.linspace(0.1, 0.9, g.n_snps * (2 if strided else 1), dtype=dtype)
    if strided:
        frequencies = frequencies[::2]
    frequencies.flags.writeable = False
    for selected in [False, True]:
        resolved = g._resolve_allele_frequencies(
            frequencies, for_selected_snps=selected
        )
        assert np.shares_memory(resolved, frequencies)
        np.testing.assert_array_equal(resolved, frequencies)


def test_sample_snapshot_reuse_and_inplace_change(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    g.filter_samples(keep_samples=["s0", "s1"])
    _ = g.maf, g.n_per_snp
    snapshot = g._sample_signature
    table = g.sample_table
    assert not np.shares_memory(snapshot, table.original_index)
    g.filter_samples(keep_samples=g.samples)
    assert g.sample_table is table
    assert g._sample_signature is snapshot
    g.sample_table.set_phenotype(np.array([0.2, 0.7]), phenotype_likelihood="gaussian")
    g.set_sample_table(g.sample_table)
    assert g._sample_signature is snapshot
    assert "MAF" in g.snp_table and "N" in g.snp_table
    # A genuine in-place row-index mutation must still invalidate cached stats,
    # even when a subsequent filter keeps every remaining row.
    g.sample_table.table.loc[g.sample_table.table.index[1], "original_index"] = 2
    np.testing.assert_array_equal(snapshot, [0, 1])
    g.filter_samples(keep_samples=g.samples)
    assert "MAF" not in g.snp_table and "N" not in g.snp_table
    np.testing.assert_allclose(g.maf, stats(known_bed[1][[0, 2]])[0], equal_nan=True)


def test_snp_table_exports_and_split_metadata_remain_independent(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    for cols in [None, ["SNP", "POS"], ["SNP", "MAF"]]:
        exported = g.get_snp_table(cols)
        exported.loc[exported.index[0], "SNP"] = "edited"
        assert g.snps[0] == "rs0"
    child = g.split_by_variants({"a": ["rs0", "rs3"]})["a"]
    child.sample_table.table.loc[child.sample_table.table.index[0], "sex"] = 2
    assert g.sample_table.table["sex"].iloc[0] == 1
    child.snp_table.loc[child.snp_table.index[0], "POS"] = 999
    assert g.bp_pos[0] == 100
    # Filtered tables must also be independent of references to the old table.
    parent_sample_table = g.sample_table.table
    parent_snp_table = g.snp_table
    g.filter_samples(keep_samples=["s0", "s1"])
    g.filter_snps(extract_snps=["rs0", "rs1"])
    g.sample_table.table.loc[g.sample_table.table.index[0], "sex"] = 2
    g.snp_table.loc[g.snp_table.index[0], "POS"] = 999
    assert parent_sample_table["sex"].iloc[0] == 1
    assert parent_snp_table["POS"].iloc[0] == 100


def test_float32_fortran_scoring_input(known_bed, tmp_path):
    g = matrix(known_bed, tmp_path)
    beta = np.asfortranarray(np.arange(g.n_snps * 2, dtype=np.float32).reshape(-1, 2))
    beta.flags.writeable = False
    np.testing.assert_allclose(g.score(beta), np.maximum(known_bed[1], 0) @ beta)
