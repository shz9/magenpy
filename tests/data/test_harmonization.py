from collections import Counter

import numpy as np
import pandas as pd
import pytest

from magenpy.AnnotationMatrix import AnnotationMatrix
from magenpy.GenotypeMatrix import GenotypeMatrix
from magenpy.GWADataLoader import GWADataLoader
from magenpy.SumstatsTable import SumstatsTable
from magenpy.utils.compute_utils import intersect_arrays, intersect_multiple_arrays
from magenpy.utils.model_utils import (
    build_snp_harmonization_plan,
    flip_allele_statistics,
    is_identity_indexer,
    merge_snp_tables,
    prepare_harmonization_plan,
)


class InMemoryGenotype:
    def __init__(self, table, genome_build=None):
        self.snp_table = table.copy()
        self.genome_build = genome_build

    @property
    def snps(self):
        return self.snp_table["SNP"].to_numpy()

    @property
    def a1(self):
        return self.snp_table["A1"].to_numpy()

    @property
    def a2(self):
        return self.snp_table["A2"].to_numpy()


class CountingLD:
    def __init__(self, snps, a1, a2, bp=None):
        self.metadata = {
            "snps": np.asarray(snps),
            "a1": np.asarray(a1),
            "a2": np.asarray(a2),
        }
        if bp is not None:
            self.metadata["bp"] = np.asarray(bp)
        self.metadata_reads = Counter()
        self._mask = None
        self.set_mask_calls = 0

    @property
    def stored_n_snps(self):
        return len(self.metadata["snps"])

    def get_metadata(self, key, apply_mask=True):
        self.metadata_reads[key] += 1
        values = self.metadata[key]
        if apply_mask and self._mask is not None:
            return values[self._mask]
        return values

    def get_mask(self):
        return self._mask

    def set_mask(self, mask):
        self.set_mask_calls += 1
        mask = np.asarray(mask)
        if mask.dtype == bool:
            self._mask = mask
        else:
            self._mask = np.zeros(self.stored_n_snps, dtype=bool)
            self._mask[mask] = True

    @property
    def snps(self):
        values = self.metadata["snps"]
        return values if self._mask is None else values[self._mask]


def make_loader(genotype=None, sumstats=None, ld=None, annotation=None):
    loader = GWADataLoader.__new__(GWADataLoader)
    loader.genotype = None if genotype is None else {1: genotype}
    loader.sumstats_table = None if sumstats is None else {1: sumstats}
    loader.ld = None if ld is None else {1: ld}
    loader.annotation = None if annotation is None else {1: annotation}
    return loader


def legacy_harmonized_sumstats(snp_ids, alleles, sumstats):
    """Reproduce the previous intersection/filter/match sequence for comparison."""

    legacy_order = [
        source
        for source in ("genotype", "sumstats", "ld", "annotation")
        if source in snp_ids
    ]
    common_snps = intersect_multiple_arrays(
        [snp_ids[source] for source in legacy_order]
    )
    filtered_positions = {
        source: intersect_arrays(snp_ids[source], common_snps, return_index=True)
        for source in legacy_order
    }

    allele_reference = "genotype" if "genotype" in snp_ids else "ld"
    ref_positions = filtered_positions[allele_reference]
    reference = pd.DataFrame(
        {
            "SNP": snp_ids[allele_reference][ref_positions],
            "A1": alleles[allele_reference][0][ref_positions],
            "A2": alleles[allele_reference][1][ref_positions],
        }
    )
    alternative = sumstats.iloc[filtered_positions["sumstats"]].reset_index(drop=True)
    return merge_snp_tables(reference, alternative)


def apply_harmonization_plan_to_sumstats(plan, sumstats):
    table = sumstats.iloc[plan["indexers"]["sumstats"]].copy().reset_index(drop=True)
    flip = plan["flip"].astype(int)
    table["BETA"] = (1.0 - 2.0 * flip) * table["BETA"]
    table["MAF"] = np.abs(flip - table["MAF"])
    table["A1"] = plan["a1"]
    table["A2"] = plan["a2"]
    return table


def test_build_harmonization_plan_returns_canonical_indexers_and_flips():
    plan = build_snp_harmonization_plan(
        {
            "reference": np.array(["rs3", "rs1", "rs2"]),
            "sumstats": np.array(["rs2", "rs1", "rs4"]),
            "annotation": np.array(["rs1", "rs2", "rs5"]),
        },
        reference="reference",
        alleles={
            "reference": (
                np.array(["G", "A", "C"]),
                np.array(["A", "G", "T"]),
            ),
            "sumstats": (
                np.array(["T", "A", "C"]),
                np.array(["C", "G", "A"]),
            ),
        },
        allele_reference="reference",
        allele_target="sumstats",
    )

    np.testing.assert_array_equal(plan["snps"], ["rs1", "rs2"])
    np.testing.assert_array_equal(plan["indexers"]["reference"], [1, 2])
    np.testing.assert_array_equal(plan["indexers"]["sumstats"], [1, 0])
    np.testing.assert_array_equal(plan["indexers"]["annotation"], [0, 1])
    np.testing.assert_array_equal(plan["flip"], [False, True])


def test_build_harmonization_plan_excludes_missing_and_duplicate_ids():
    plan = build_snp_harmonization_plan(
        {
            "reference": np.array(["rs1", "rs2", "rs2", None, "rs3"], dtype=object),
            "other": np.array(["rs3", "rs1", "rs2"], dtype=object),
        },
        reference="reference",
    )

    np.testing.assert_array_equal(plan["snps"], ["rs1", "rs3"])
    np.testing.assert_array_equal(plan["indexers"]["reference"], [0, 4])
    np.testing.assert_array_equal(plan["indexers"]["other"], [1, 0])


def test_build_harmonization_plan_accepts_composite_variant_ids():
    plan = build_snp_harmonization_plan(
        {
            "reference": pd.MultiIndex.from_tuples([(1, 20), (1, 10)]),
            "other": pd.MultiIndex.from_tuples([(1, 10), (1, 30), (1, 20)]),
        },
        reference="reference",
    )

    assert plan["variant_ids"].tolist() == [(1, 20), (1, 10)]
    np.testing.assert_array_equal(plan["indexers"]["reference"], [0, 1])
    np.testing.assert_array_equal(plan["indexers"]["other"], [2, 0])


def test_genotype_duplicate_filter_prefers_positions_over_snp_ids():
    genotype = GenotypeMatrix.__new__(GenotypeMatrix)
    genotype.snp_table = pd.DataFrame(
        {
            "CHR": [1, 1, 1, 1],
            "POS": [10, 20, 30, 30],
            "SNP": [".", ".", "rs3a", "rs3b"],
        }
    )

    genotype.drop_duplicated_snps()

    np.testing.assert_array_equal(genotype.snp_table["POS"], [10, 20])


def test_identity_indexer_handles_aligned_and_reordered_positions():
    assert is_identity_indexer(np.arange(200_000, dtype=np.int32), 200_000)
    assert is_identity_indexer(np.arange(200_000, dtype=np.int64), 200_000)
    assert is_identity_indexer(np.array([], dtype=int), 0)
    assert not is_identity_indexer(np.array([0, 2, 1]), 3)
    assert not is_identity_indexer(np.array([0, 1]), 3)
    assert not is_identity_indexer(np.arange(10)[::2], 5)


def test_native_identity_indexer_checks_int32_and_int64_arrays():
    variant_cpp = pytest.importorskip("magenpy.stats.variant.variant_cpp")
    if not hasattr(variant_cpp, "is_identity_indexer_native"):
        pytest.skip("The variant extension must be rebuilt from Cython.")

    for dtype in (np.int32, np.int64):
        indexer = np.arange(100_000, dtype=dtype)
        assert variant_cpp.is_identity_indexer_native(indexer, len(indexer))
        indexer[-1] = 0
        assert not variant_cpp.is_identity_indexer_native(indexer, len(indexer))


@pytest.mark.parametrize(
    "source_names",
    [
        ("genotype", "sumstats"),
        ("sumstats", "ld"),
        ("genotype", "sumstats", "ld", "annotation"),
    ],
)
def test_harmonization_matches_legacy_results_under_random_snp_orderings(
    source_names,
):
    """Compare variant membership and allele corrections with the old algorithm."""

    bases = np.array(["A", "C", "G", "T"])

    for seed in range(25):
        rng = np.random.default_rng(seed)
        n_variants = 80
        all_snps = np.array([f"rs{i}" for i in range(n_variants)])
        allele_index = rng.integers(0, len(bases), size=n_variants)
        canonical_a1 = bases[allele_index]
        canonical_a2 = bases[
            (allele_index + rng.integers(1, len(bases), size=n_variants)) % len(bases)
        ]

        snp_ids = {}
        alleles = {}
        source_global_positions = {}
        for source in source_names:
            n_source = rng.integers(55, n_variants + 1)
            positions = rng.permutation(n_variants)[:n_source]
            source_global_positions[source] = positions
            snp_ids[source] = all_snps[positions]
            if source in ("genotype", "sumstats", "ld"):
                alleles[source] = (
                    canonical_a1[positions].copy(),
                    canonical_a2[positions].copy(),
                )

        sumstats_positions = source_global_positions["sumstats"]
        sumstats_a1, sumstats_a2 = alleles["sumstats"]
        flip = rng.random(len(sumstats_positions)) < 0.2
        sumstats_a1[flip], sumstats_a2[flip] = (
            sumstats_a2[flip].copy(),
            sumstats_a1[flip].copy(),
        )

        # Introduce allele-incompatible variants as well as ordinary flips.
        incompatible = rng.random(len(sumstats_positions)) < 0.1
        sumstats_a1[incompatible] = "N"
        sumstats_a2[incompatible] = "N"
        sumstats = pd.DataFrame(
            {
                "SNP": snp_ids["sumstats"],
                "A1": sumstats_a1,
                "A2": sumstats_a2,
                "BETA": rng.normal(size=len(sumstats_positions)),
                "MAF": rng.random(len(sumstats_positions)),
            }
        )

        order_reference = "ld" if "ld" in source_names else "genotype"
        allele_reference = "genotype" if "genotype" in source_names else "ld"
        plan = build_snp_harmonization_plan(
            snp_ids,
            reference=order_reference,
            alleles=alleles,
            allele_reference=allele_reference,
            allele_target="sumstats",
        )

        # Every source must resolve to exactly the same ordered SNP vector,
        # regardless of its input ordering.
        for source, indexer in plan["indexers"].items():
            np.testing.assert_array_equal(snp_ids[source][indexer], plan["snps"])

        legacy = legacy_harmonized_sumstats(snp_ids, alleles, sumstats)
        streamlined = apply_harmonization_plan_to_sumstats(plan, sumstats)
        compare_columns = ["SNP", "A1", "A2", "BETA", "MAF"]
        legacy = legacy[compare_columns].sort_values("SNP").reset_index(drop=True)
        streamlined = (
            streamlined[compare_columns].sort_values("SNP").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(streamlined, legacy, check_dtype=False)


def test_harmonize_data_uses_one_ld_snp_read_and_one_final_mask():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3", "rs5"],
                "A1": ["A", "C", "G", "A"],
                "A2": ["G", "T", "A", "C"],
                "original_index": [0, 1, 2, 3],
            }
        )
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1, 1, 1],
                "SNP": ["rs2", "rs3", "rs1", "rs6"],
                "A1": ["T", "T", "A", "A"],
                "A2": ["C", "C", "G", "G"],
                "BETA": [2.0, 3.0, 1.0, 6.0],
                "MAF": [0.2, 0.3, 0.1, 0.4],
            }
        )
    )
    ld = CountingLD(
        ["rs3", "rs1", "rs2", "rs4"],
        ["G", "A", "C", "T"],
        ["A", "G", "T", "C"],
    )
    annotation = AnnotationMatrix(
        pd.DataFrame(
            {
                "CHR": [1, 1, 1, 1],
                "SNP": ["rs2", "rs1", "rs3", "rs7"],
                "POS": [20, 10, 30, 70],
                "coding": [0, 1, 0, 1],
            }
        )
    )
    loader = make_loader(genotype, sumstats, ld, annotation)

    loader.harmonize_data()

    expected_snps = np.array(["rs1", "rs2"])
    np.testing.assert_array_equal(genotype.snps, expected_snps)
    np.testing.assert_array_equal(sumstats.snps, expected_snps)
    np.testing.assert_array_equal(annotation.snps, expected_snps)
    np.testing.assert_array_equal(ld.snps, expected_snps)
    np.testing.assert_array_equal(sumstats.table["BETA"], [1.0, -2.0])
    np.testing.assert_allclose(sumstats.table["MAF"], [0.1, 0.8])

    assert ld.metadata_reads == {"snps": 1}
    assert ld.set_mask_calls == 1


def test_harmonize_data_reads_each_ld_allele_once_when_ld_is_reference():
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "T"],
                "A2": ["G", "C"],
                "BETA": [1.0, 2.0],
            }
        )
    )
    ld = CountingLD(
        ["rs2", "rs1", "rs3"],
        ["C", "A", "G"],
        ["T", "G", "A"],
    )
    loader = make_loader(sumstats=sumstats, ld=ld)

    loader.harmonize_data()

    np.testing.assert_array_equal(sumstats.snps, ["rs2", "rs1"])
    np.testing.assert_array_equal(sumstats.table["BETA"], [-2.0, 1.0])
    assert ld.metadata_reads == {"snps": 1, "a1": 1, "a2": 1}
    assert ld.set_mask_calls == 1


def test_harmonize_data_uses_ld_positions_when_snp_ids_are_missing():
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "POS": [10, 20],
                "A1": ["A", "T"],
                "A2": ["G", "C"],
                "BETA": [1.0, 2.0],
            }
        )
    )
    ld = CountingLD(
        ["ld2", "ld1", "ld3"],
        ["C", "A", "G"],
        ["T", "G", "A"],
        bp=[20, 10, 30],
    )

    make_loader(sumstats=sumstats, ld=ld).harmonize_data()

    np.testing.assert_array_equal(sumstats.table["POS"], [20, 10])
    np.testing.assert_array_equal(sumstats.table["BETA"], [-2.0, 1.0])
    assert ld.metadata_reads == {"bp": 1, "a1": 1, "a2": 1}


def test_harmonize_data_reuses_aligned_sumstats_dataframe():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "C"],
                "A2": ["G", "T"],
            }
        )
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "C"],
                "A2": ["G", "T"],
                "BETA": [1.0, 2.0],
            }
        )
    )
    original_table = sumstats.table

    make_loader(genotype=genotype, sumstats=sumstats).harmonize_data()

    assert sumstats.table is original_table


def test_harmonize_data_prefers_position_and_does_not_require_snp_ids():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "POS": [20, 10],
                "SNP": ["genotype_2", "genotype_1"],
                "A1": ["C", "A"],
                "A2": ["T", "G"],
            }
        )
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "POS": [10, 20],
                "A1": ["A", "T"],
                "A2": ["G", "C"],
                "BETA": [1.0, 2.0],
            }
        )
    )
    annotation = AnnotationMatrix(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "POS": [10, 20],
                "coding": [1, 0],
            }
        )
    )

    make_loader(genotype, sumstats, annotation=annotation).harmonize_data()

    np.testing.assert_array_equal(sumstats.table["POS"], [20, 10])
    np.testing.assert_array_equal(sumstats.table["BETA"], [-2.0, 1.0])
    np.testing.assert_array_equal(annotation.table["POS"], [20, 10])


def test_harmonize_data_falls_back_to_snp_ids_without_positions():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs2", "rs1"],
                "A1": ["C", "A"],
                "A2": ["T", "G"],
            }
        )
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "CHR": [1, 1],
                "POS": [10, 20],
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "C"],
                "A2": ["G", "T"],
                "BETA": [1.0, 2.0],
            }
        )
    )

    make_loader(genotype, sumstats).harmonize_data()

    np.testing.assert_array_equal(sumstats.table["SNP"], ["rs2", "rs1"])


def test_harmonize_data_rejects_conflicting_genome_builds():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs1"],
                "A1": ["A"],
                "A2": ["G"],
            }
        ),
        genome_build="GRCh37",
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": ["rs1"],
                "A1": ["A"],
                "A2": ["G"],
                "BETA": [1.0],
            }
        ),
        genome_build="GRCh38",
    )

    with pytest.raises(ValueError, match="conflicting genome builds"):
        make_loader(genotype, sumstats).harmonize_data()


def test_harmonize_data_accepts_equivalent_genome_build_names():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs1"],
                "A1": ["A"],
                "A2": ["G"],
            }
        ),
        genome_build="hg19",
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": ["rs1"],
                "A1": ["A"],
                "A2": ["G"],
                "BETA": [1.0],
            }
        ),
        genome_build="GRCh37",
    )

    make_loader(genotype, sumstats).harmonize_data()


@pytest.mark.parametrize("path", ["merge", "loader"])
def test_shared_allele_correction_updates_all_statistics(path):
    reference = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
        }
    )
    alternative = pd.DataFrame(
        {
            "SNP": ["rs3", "rs2", "rs1"],
            "A1": ["G", "T", "A"],
            "A2": ["A", "C", "G"],
            "BETA": [3.0, 2.0, 1.0],
            "STD_BETA": [6.0, 4.0, 2.0],
            "Z": [9.0, 6.0, 3.0],
            "MAF": [0.3, 0.2, 0.1],
            "SE": [0.03, 0.02, 0.01],
            "PVAL": [0.003, 0.002, 0.001],
            "OR": [1.3, 1.2, 1.1],
        },
        index=[40, 40, 10],
    )
    original = alternative.copy()
    if path == "merge":
        result = merge_snp_tables(reference, alternative)
    else:
        sumstats = SumstatsTable(alternative.copy())
        make_loader(InMemoryGenotype(reference), sumstats).harmonize_data()
        result = sumstats.table
    np.testing.assert_array_equal(result["SNP"], ["rs1", "rs2", "rs3"])
    for column, expected in {
        "BETA": [1.0, -2.0, 3.0],
        "STD_BETA": [2.0, -4.0, 6.0],
        "Z": [3.0, -6.0, 9.0],
        "MAF": [0.1, 0.8, 0.3],
        "SE": [0.01, 0.02, 0.03],
        "PVAL": [0.001, 0.002, 0.003],
        "OR": [1.1, 1.2, 1.3],
    }.items():
        np.testing.assert_allclose(result[column], expected)
    np.testing.assert_array_equal(result["A1"], reference["A1"])
    np.testing.assert_array_equal(result["A2"], reference["A2"])
    pd.testing.assert_frame_equal(alternative, original)


def test_flip_statistics_is_positional_and_in_place():
    table = pd.DataFrame(
        {
            "CUSTOM": [1.0, 2.0, 3.0],
            "BETA": [4.0, 5.0, 6.0],
            "MAF": [0.1, 0.2, np.nan],
            "A1": ["A", "C", "G"],
        },
        index=[20, 20, 5],
    )
    result = flip_allele_statistics(
        table, [False, True, True], signed_statistics="CUSTOM"
    )
    assert result is table
    np.testing.assert_array_equal(table["CUSTOM"], [1.0, -2.0, -3.0])
    np.testing.assert_array_equal(table["BETA"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(table["MAF"], [0.1, 0.8, np.nan], equal_nan=True)
    np.testing.assert_array_equal(table["A1"], ["A", "C", "G"])
    np.testing.assert_array_equal(table.index, [20, 20, 5])


@pytest.mark.parametrize("mask", [[], [True], [[True, False]]])
def test_flip_statistics_rejects_misaligned_masks_before_mutation(mask):
    table = pd.DataFrame({"BETA": [1.0, 2.0]})
    original = table.copy()
    with pytest.raises(ValueError, match="one-dimensional mask"):
        flip_allele_statistics(table, mask)
    pd.testing.assert_frame_equal(table, original)


def test_flip_statistics_handles_empty_and_unflipped_tables():
    for table, mask in [
        (pd.DataFrame({"BETA": pd.Series(dtype=float)}), []),
        (pd.DataFrame({"BETA": [1.0, np.nan]}), [False, False]),
        (pd.DataFrame({"SE": [0.1, 0.2]}), [True, False]),
    ]:
        original = table.copy()
        assert flip_allele_statistics(table, mask) is table
        pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("correct_flips", [True, False])
def test_merge_left_join_preserves_unmatched_rows_and_custom_statistics(correct_flips):
    reference = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
        }
    )
    alternative = pd.DataFrame(
        {
            "SNP": ["rs1"],
            "A1": ["G"],
            "A2": ["A"],
            "CUSTOM": [2.0],
            "BETA": [3.0],
            "MAF": [0.2],
        }
    )
    result = merge_snp_tables(
        reference,
        alternative,
        how="left",
        signed_statistics="CUSTOM",
        correct_flips=correct_flips,
    )
    np.testing.assert_array_equal(result["SNP"], ["rs1", "rs2"])
    assert result["CUSTOM"].iloc[0] == (-2.0 if correct_flips else 2.0)
    assert result["BETA"].iloc[0] == 3.0
    assert result["MAF"].iloc[0] == (0.8 if correct_flips else 0.2)
    assert pd.isna(result["CUSTOM"].iloc[1])
    np.testing.assert_array_equal(result["A1"], reference["A1"])


def test_harmonization_plan_preserves_existing_ld_mask_and_is_idempotent():
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": ["rs3", "rs1"],
                "A1": ["T", "A"],
                "A2": ["C", "G"],
                "BETA": [2.0, 1.0],
                "MAF": [0.2, 0.1],
            }
        )
    )
    ld = CountingLD(
        ["rs1", "rs2", "rs3", "rs4"], ["A", "G", "C", "A"], ["G", "A", "T", "C"]
    )
    ld.set_mask([True, False, True, True])
    loader = make_loader(sumstats=sumstats, ld=ld)
    plan, positions = prepare_harmonization_plan(1, {"sumstats": sumstats, "ld": ld})
    np.testing.assert_array_equal(positions, [0, 2, 3])
    np.testing.assert_array_equal(ld.get_mask(), [True, False, True, True])
    assert ld.set_mask_calls == 1  # Planning never changes the source.
    np.testing.assert_array_equal(sumstats.snps, ["rs3", "rs1"])
    loader.harmonize_data()
    np.testing.assert_array_equal(ld.get_mask(), [True, False, True, False])
    np.testing.assert_array_equal(sumstats.snps, ["rs1", "rs3"])
    np.testing.assert_array_equal(sumstats.table["BETA"], [1.0, -2.0])
    expected = sumstats.table.copy()
    calls = ld.set_mask_calls
    loader.harmonize_data()
    pd.testing.assert_frame_equal(sumstats.table, expected)
    assert ld.set_mask_calls == calls


def test_harmonization_removes_chromosomes_missing_from_any_source():
    reference = pd.DataFrame({"SNP": ["rs1"], "A1": ["A"], "A2": ["G"]})
    loader = make_loader(InMemoryGenotype(reference), SumstatsTable(reference.copy()))
    loader.genotype[2] = InMemoryGenotype(reference)
    loader.sumstats_table[3] = SumstatsTable(reference.copy())
    loader.harmonize_data()
    assert set(loader.genotype) == set(loader.sumstats_table) == {1}


def test_harmonization_handles_empty_variant_intersection():
    genotype = InMemoryGenotype(
        pd.DataFrame(
            {
                "SNP": ["rs1"],
                "A1": ["A"],
                "A2": ["G"],
            }
        )
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": ["rs2"],
                "A1": ["C"],
                "A2": ["T"],
                "BETA": [1.0],
            }
        )
    )
    ld = CountingLD(["rs1", "rs2"], ["A", "C"], ["G", "T"])
    make_loader(genotype, sumstats, ld).harmonize_data()
    assert len(genotype.snp_table) == len(sumstats.table) == 0
    np.testing.assert_array_equal(ld.get_mask(), [False, False])
    assert ld.set_mask_calls == 1


def test_harmonization_selects_xarray_genotypes_with_their_metadata(tmp_path):
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")
    from magenpy.GenotypeMatrix import xarrayGenotypeMatrix

    values = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    genotype = xarrayGenotypeMatrix(
        sample_table=pd.DataFrame({"FID": ["F", "F"], "IID": ["s1", "s2"]}),
        snp_table=pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3"],
                "A1": ["A", "C", "G"],
                "A2": ["G", "T", "A"],
            }
        ),
        xr_mat=xr.DataArray(
            da.from_array(values),
            dims=("sample", "variant"),
            coords={"sample": ["s1", "s2"]},
        ),
        temp_dir=str(tmp_path),
    )
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": ["rs3", "rs2"],
                "A1": ["G", "C"],
                "A2": ["A", "T"],
            }
        )
    )
    make_loader(genotype, sumstats).harmonize_data()
    np.testing.assert_array_equal(genotype.snps, ["rs2", "rs3"])
    np.testing.assert_array_equal(genotype.to_numpy(), values[:, [1, 2]])


@pytest.mark.parametrize("premasked", [False, True])
@pytest.mark.parametrize("remove_variants", [False, True])
def test_harmonization_uses_ld_index_api_only_when_selection_changes(
    tmp_path, monkeypatch, premasked, remove_variants
):
    from unittest.mock import Mock
    from scipy.sparse import csr_matrix
    from magenpy.LDMatrix import LDMatrix

    ld = LDMatrix.from_csr(csr_matrix((4, 4)), str(tmp_path / "ld"), dtype="float32")
    snps = np.array(["rs1", "rs2", "rs3", "rs4"])
    ld.set_metadata("snps", snps)
    ld.set_metadata("a1", np.array(["A"] * 4))
    ld.set_metadata("a2", np.array(["G"] * 4))
    if premasked:
        ld.set_mask(np.array([True, False, True, True]))
    original_mask = ld.get_mask()
    visible = np.flatnonzero(original_mask) if premasked else np.arange(4)
    retained = visible[::2] if remove_variants else visible
    sumstats = SumstatsTable(
        pd.DataFrame(
            {
                "SNP": snps[retained[::-1]],
                "A1": "A",
                "A2": "G",
            }
        )
    )
    setter = Mock(wraps=ld.set_mask)
    monkeypatch.setattr(ld, "set_mask", setter)
    loader = make_loader(sumstats=sumstats, ld=ld)
    loader.harmonize_data()
    np.testing.assert_array_equal(ld.snps, snps[retained])
    np.testing.assert_array_equal(sumstats.snps, snps[retained])
    if remove_variants:
        setter.assert_called_once()
        indices = setter.call_args.args[0]
        assert np.issubdtype(indices.dtype, np.integer)
        np.testing.assert_array_equal(indices, retained)
    else:
        setter.assert_not_called()
        assert ld.get_mask() is original_mask
    calls = setter.call_count
    loader.harmonize_data()
    assert setter.call_count == calls
