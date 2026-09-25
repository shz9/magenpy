from collections import Counter

import numpy as np
import pandas as pd
import pytest

from magenpy.AnnotationMatrix import AnnotationMatrix
from magenpy.GWADataLoader import GWADataLoader
from magenpy.SumstatsTable import SumstatsTable
from magenpy.utils.compute_utils import intersect_arrays, intersect_multiple_arrays
from magenpy.utils.model_utils import (
    build_snp_harmonization_plan,
    merge_snp_tables,
)


class InMemoryGenotype:
    def __init__(self, table):
        self.snp_table = table.copy()

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
    def __init__(self, snps, a1, a2):
        self.metadata = {
            "snps": np.asarray(snps),
            "a1": np.asarray(a1),
            "a2": np.asarray(a2),
        }
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
        self._mask = np.asarray(mask, dtype=bool)

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
        source for source in ("genotype", "sumstats", "ld", "annotation")
        if source in snp_ids
    ]
    common_snps = intersect_multiple_arrays([snp_ids[source] for source in legacy_order])
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
            (allele_index + rng.integers(1, len(bases), size=n_variants))
            % len(bases)
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
