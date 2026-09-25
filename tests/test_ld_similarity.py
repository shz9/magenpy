import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import magenpy as mgp
from magenpy.stats.ld import compute_ld_sumstats_similarity
from magenpy.stats.ld.similarity import _load_ld_block


def _make_ld(
    tmp_path,
    name,
    correlation,
    frequencies,
    a1=None,
    a2=None,
    chromosome=1,
    dtype="float64",
):
    n_variants = correlation.shape[0]
    ld = mgp.LDMatrix.from_csr(
        csr_matrix(correlation),
        store_path=str(tmp_path / name),
        overwrite=True,
        dtype=dtype,
    )
    ld.set_metadata("snps", np.array([f"rs{i}" for i in range(n_variants)]))
    ld.set_metadata("a1", np.full(n_variants, "A") if a1 is None else a1)
    ld.set_metadata("a2", np.full(n_variants, "G") if a2 is None else a2)
    ld.set_metadata("bp", np.arange(n_variants) * 1_000 + 1)
    if frequencies is not None:
        ld.set_metadata("maf", np.asarray(frequencies))
    ld.set_store_attr("Chromosome", chromosome)
    return ld


def test_load_ld_block_uses_low_precision_for_float32_store(tmp_path, monkeypatch):
    n_variants = 8
    index = np.arange(n_variants)
    correlation = 0.6 ** np.abs(index[:, None] - index[None, :])
    ld = _make_ld(
        tmp_path,
        "float32_masked",
        correlation,
        None,
        dtype="float32",
    )
    ld.set_mask(np.array([0, 2, 4, 5, 7]))

    requested_dtypes = []
    load_data = ld.load_data

    def record_dtype(*args, **kwargs):
        requested_dtypes.append(np.dtype(kwargs["dtype"]))
        return load_data(*args, **kwargs)

    monkeypatch.setattr(ld, "load_data", record_dtype)
    selected = np.array([0, 4, 7])
    block = _load_ld_block(ld, selected, np.flatnonzero(ld.get_mask()))

    assert requested_dtypes == [np.dtype("float32")]
    assert block.dtype == np.dtype("float64")
    np.testing.assert_allclose(
        block,
        correlation[np.ix_(selected, selected)],
        rtol=1e-6,
        atol=1e-7,
    )


def test_frequency_similarity_returns_relative_probabilities(tmp_path):
    n_variants = 30
    frequencies = np.linspace(0.08, 0.48, n_variants)
    identity = np.eye(n_variants)
    matched = _make_ld(tmp_path, "matched", identity, frequencies + 0.005)
    mismatched = _make_ld(
        tmp_path, "mismatched", identity, np.clip(frequencies + 0.16, 0.0, 1.0)
    )
    sumstats = pd.DataFrame(
        {
            "CHR": 1,
            "SNP": [f"rs{i}" for i in range(n_variants)],
            "POS": np.arange(n_variants) * 1_000 + 1,
            "A1": "A",
            "A2": "G",
            "MAF": frequencies,
        }
    )

    result = compute_ld_sumstats_similarity(
        sumstats,
        {"matched": matched, "mismatched": mismatched},
        block_size=4,
        n_blocks=3,
        min_variants=10,
    )

    assert result.index[0] == "matched"
    assert result.loc["matched", "frequency_rmse"] < result.loc[
        "mismatched", "frequency_rmse"
    ]
    assert result.loc["matched", "probability"] > 0.9
    assert np.isclose(result["probability"].sum(), 1.0)
    assert set(result["method"]) == {"frequency"}
    assert set(result["n_blocks"]) == {3}
    assert set(result["n_variants"]) == {12}


def test_frequency_similarity_corrects_swapped_alleles(tmp_path):
    n_variants = 24
    frequencies = np.linspace(0.1, 0.4, n_variants)
    swapped = _make_ld(
        tmp_path,
        "swapped",
        np.eye(n_variants),
        1.0 - frequencies,
        a1=np.full(n_variants, "G"),
        a2=np.full(n_variants, "A"),
    )
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i}" for i in range(n_variants)],
            "A1": "A",
            "A2": "G",
            "MAF": frequencies,
        }
    )

    result = compute_ld_sumstats_similarity(
        sumstats, [swapped], labels=["swapped"], method="frequency"
    )

    assert result.loc["swapped", "frequency_rmse"] < 1e-7
    assert result.loc["swapped", "probability"] == 1.0


def test_frequency_similarity_matches_by_position_without_snp_ids(tmp_path):
    n_variants = 24
    frequencies = np.linspace(0.1, 0.4, n_variants)
    ld = _make_ld(tmp_path, "position_only", np.eye(n_variants), frequencies)
    sumstats = pd.DataFrame(
        {
            "CHR": 1,
            "POS": np.arange(n_variants) * 1_000 + 1,
            "A1": "A",
            "A2": "G",
            "MAF": frequencies,
            "UNUSED": np.arange(n_variants),
        }
    )
    original = sumstats.copy()

    result = compute_ld_sumstats_similarity(
        sumstats, [ld], labels=["position_only"], method="frequency"
    )

    assert result.loc["position_only", "frequency_rmse"] < 1e-7
    pd.testing.assert_frame_equal(sumstats, original)


def test_ld_likelihood_prefers_generating_correlation(tmp_path):
    rng = np.random.default_rng(42)
    n_variants = 120
    index = np.arange(n_variants)
    matched_correlation = 0.75 ** np.abs(index[:, None] - index[None, :])
    mismatched_correlation = 0.05 ** np.abs(index[:, None] - index[None, :])
    z_scores = rng.multivariate_normal(np.zeros(n_variants), matched_correlation)

    matched = _make_ld(tmp_path, "matched_ld", matched_correlation, None)
    mismatched = _make_ld(
        tmp_path, "mismatched_ld", mismatched_correlation, None
    )
    sumstats = mgp.SumstatsTable(
        pd.DataFrame(
            {
                "CHR": 1,
                "SNP": [f"rs{i}" for i in range(n_variants)],
                "POS": index * 1_000 + 1,
                "A1": "A",
                "A2": "G",
                "Z": z_scores,
            }
        )
    )

    result = compute_ld_sumstats_similarity(
        sumstats,
        [matched, mismatched],
        labels=["matched", "mismatched"],
        block_size=40,
        n_blocks=3,
    )

    assert result.index[0] == "matched"
    assert result.loc["matched", "score"] > result.loc["mismatched", "score"]
    assert np.isclose(result["probability"].sum(), 1.0)
    assert set(result["method"]) == {"ld_likelihood"}
    assert set(result["n_blocks"]) == {3}
    assert set(result["n_variants"]) == {120}


def test_genome_wide_similarity_accepts_harmonized_data_loaders(tmp_path):
    rng = np.random.default_rng(123)
    n_variants = 60
    index = np.arange(n_variants)
    matched_correlation = 0.7 ** np.abs(index[:, None] - index[None, :])
    mismatched_correlation = 0.05 ** np.abs(index[:, None] - index[None, :])

    matched_gdl = mgp.GWADataLoader(temp_dir=str(tmp_path / "matched_temp"))
    mismatched_gdl = mgp.GWADataLoader(temp_dir=str(tmp_path / "mismatched_temp"))
    matched_gdl.ld, mismatched_gdl.ld = {}, {}
    matched_gdl.sumstats_table, mismatched_gdl.sumstats_table = {}, {}

    for chromosome in (1, 2):
        z_scores = rng.multivariate_normal(
            np.zeros(n_variants), matched_correlation
        )
        matched_gdl.ld[chromosome] = _make_ld(
            tmp_path,
            f"matched_chr{chromosome}",
            matched_correlation,
            None,
            chromosome=chromosome,
        )
        mismatched_gdl.ld[chromosome] = _make_ld(
            tmp_path,
            f"mismatched_chr{chromosome}",
            mismatched_correlation,
            None,
            chromosome=chromosome,
        )
        table = pd.DataFrame(
            {
                "CHR": chromosome,
                "SNP": [f"rs{i}" for i in range(n_variants)],
                "POS": index * 1_000 + 1,
                "A1": "A",
                "A2": "G",
                "Z": z_scores,
            }
        )
        matched_gdl.sumstats_table[chromosome] = mgp.SumstatsTable(table.copy())
        mismatched_gdl.sumstats_table[chromosome] = mgp.SumstatsTable(table.copy())

    result = compute_ld_sumstats_similarity(
        data_loaders=[matched_gdl, mismatched_gdl],
        labels=["matched", "mismatched"],
        method="ld_likelihood",
        block_size=20,
        n_blocks=4,
        min_variants=60,
    )

    assert result.index[0] == "matched"
    assert set(result["n_blocks"]) == {4}
    assert set(result["n_variants"]) == {80}
