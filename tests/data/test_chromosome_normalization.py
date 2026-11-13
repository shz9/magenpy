"""Chromosome identities must agree before source dictionaries are harmonized."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from magenpy.AnnotationMatrix import AnnotationMatrix
from magenpy.GenotypeMatrix import MagenpyGenotypeMatrix, plinkBEDGenotypeMatrix
from magenpy.GWADataLoader import GWADataLoader
from magenpy.LDMatrix import LDMatrix
from magenpy.SumstatsTable import SumstatsTable
from magenpy.parsers.annotation_parsers import (
    AnnotationMatrixParser,
    parse_annotation_bed_file,
)
from magenpy.parsers.misc_parsers import parse_ld_block_data
from magenpy.parsers.plink_parsers import parse_bim_file
from magenpy.parsers.sumstats_parsers import SumstatsParser, Plink2SSParser
from magenpy.utils.chromosome_utils import (
    normalize_chromosome,
    normalize_chromosome_column,
)
from magenpy.utils.model_utils import match_chromosomes


@pytest.mark.parametrize(
    "label,expected",
    [
        (1, 1),
        (np.int32(1), 1),
        (1.0, 1),
        ("1", 1),
        ("01", 1),
        ("1.0", 1),
        ("chr1", 1),
        ("chr:1", 1),
        ("chr_1", 1),
        (" CHR:01 ", 1),
        ("X", 23),
        ("chr:x", 23),
        ("23", 23),
        ("chrY", 24),
        ("XY", 25),
        ("M", 26),
        ("chrMT", 26),
        ("26", 26),
        (b"chr1", 1),
        ("chrUn_KI270442v1", "Un_KI270442v1"),
        ("GL000220.1", "GL000220.1"),
        ("mychr1", "mychr1"),
    ],
)
def test_canonical_labels(label, expected):
    assert normalize_chromosome(label) == expected


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "chr:"])
def test_missing_chromosome_is_not_an_identifier(missing):
    assert pd.isna(normalize_chromosome(missing))


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_column_normalization_handles_aliases_and_missing_values(dtype):
    table = pd.DataFrame(
        {
            "CHR": pd.Series(["chr1", "1", "chr:1", "chrX", None], dtype=dtype),
            "value": np.arange(5),
        }
    )
    result = normalize_chromosome_column(table)
    assert result is table
    assert list(result["CHR"].dropna()) == [1, 1, 1, 23]
    assert str(result["CHR"].dtype) == "category"
    np.testing.assert_array_equal(result["CHR"].cat.categories, np.arange(27))
    np.testing.assert_array_equal(result["CHR"].cat.codes, [1, 1, 1, 23, -1])
    assert pd.isna(result["CHR"].iloc[-1])
    np.testing.assert_array_equal(result["value"], np.arange(5))


def test_canonical_categorical_column_reuses_compact_codes():
    table = pd.DataFrame({"CHR": np.arange(1, 24, dtype=np.int32)})
    normalize_chromosome_column(table)
    original = table["CHR"].array
    assert original.codes.dtype == np.int8
    normalize_chromosome_column(table)
    assert table["CHR"].array is original


def test_matching_normalizes_every_label_even_with_partial_literal_matches():
    left, right = match_chromosomes(
        [1, 2, 3], ["1", "chr:2", "CHR_3"], return_both=True
    )
    np.testing.assert_array_equal(left, [1, 2, 3])
    np.testing.assert_array_equal(right, ["1", "chr:2", "CHR_3"])
    left, right = match_chromosomes(
        ["chrX", "mychr1", None], [23, "my1", None], return_both=True
    )
    np.testing.assert_array_equal(left, ["chrX"])
    np.testing.assert_array_equal(right, [23])


@pytest.fixture
def prefixed_bed(tmp_path):
    prefix = tmp_path / "genotypes"
    prefix.with_suffix(".fam").write_text("F s1 0 0 1 -9\nF s2 0 0 2 -9\n")
    prefix.with_suffix(".bim").write_text(
        "chr1\trs1\t0\t10\tA\tG\n"
        "chr:1\trs2\t0\t20\tC\tT\n"
        "CHR_2\trs3\t0\t30\tA\tC\n"
        "chrX\trs4\t0\t40\tG\tT\n"
    )
    prefix.with_suffix(".bed").write_bytes(b"\x6c\x1b\x01" + bytes([12, 2, 8, 14]))
    return str(prefix)


@pytest.mark.parametrize("backend", ["magenpy", "plink", "bed-reader"])
def test_genotype_readers_normalize_before_splitting(prefixed_bed, tmp_path, backend):
    classes = {"magenpy": MagenpyGenotypeMatrix, "plink": plinkBEDGenotypeMatrix}
    if backend == "bed-reader":
        pytest.importorskip("bed_reader")
        from magenpy.GenotypeMatrix import bedReaderGenotypeMatrix

        classes[backend] = bedReaderGenotypeMatrix
    parsed = parse_bim_file(prefixed_bed)
    np.testing.assert_array_equal(parsed["CHR"], [1, 1, 2, 23])
    g = classes[backend].from_file(prefixed_bed, temp_dir=str(tmp_path))
    split = g.split_by_chromosome()
    assert set(split) == {1, 2, 23}
    np.testing.assert_array_equal(split[1].snps, ["rs1", "rs2"])
    if backend != "plink":
        expected = np.array([[2, 1, 2, 1], [0, 2, 1, 0]], dtype=np.int8)
        np.testing.assert_array_equal(g.to_numpy(), expected)
        np.testing.assert_array_equal(split[1].to_numpy(), expected[:, :2])


@pytest.mark.parametrize(
    "parser,header",
    [
        (SumstatsParser, "CHR\tPOS\tA1\tA2\tBETA\tSE\n"),
        (Plink2SSParser, "#CHROM\tPOS\tA1\tREF\tBETA\tSE\n"),
    ],
)
def test_summary_parsers_normalize_after_column_renaming(tmp_path, parser, header):
    path = tmp_path / "sumstats.tsv"
    path.write_text(header + "chr1\t10\tA\tG\t.1\t.01\nchr:1\t20\tC\tT\t.2\t.02\n")
    table = parser(sep="\t").parse(path)
    np.testing.assert_array_equal(table["CHR"], [1, 1])
    assert set(SumstatsTable(table).split_by_chromosome()) == {1}


def test_bed_annotation_and_ld_block_parsers_merge_aliases(tmp_path):
    path = tmp_path / "intervals.bed"
    path.write_text("chr1\t10\t20\nchr:1\t30\t40\nchrX\t50\t60\n")
    np.testing.assert_array_equal(parse_annotation_bed_file(path)["CHR"], [1, 1, 23])
    path.write_text("chr start stop\nchr1 10 20\nchr:1 30 40\nchrX 50 60\n")
    blocks = parse_ld_block_data(path)
    assert set(blocks) == {1, 23}
    np.testing.assert_array_equal(blocks[1], [[10, 20], [30, 40]])


def test_loader_harmonizes_differently_encoded_files_before_matching(
    prefixed_bed, tmp_path
):
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "CHR\tPOS\tA1\tA2\tBETA\tSE\n"
        "1\t10\tA\tG\t.1\t.01\n"
        "CHR:1\t20\tT\tC\t.2\t.02\n"
        "2\t30\tA\tC\t.3\t.03\n"
        "23\t40\tG\tT\t.4\t.04\n"
    )
    annotations = tmp_path / "annotations.annot"
    annotations.write_text(
        "CHR\tPOS\tcoding\nchr_1\t10\t1\n1\t20\t0\nchr:2\t30\t1\nX\t40\t0\n"
    )
    loader = GWADataLoader(
        bed_files=prefixed_bed,
        sumstats_files=str(summary),
        annotation_files=str(annotations),
        temp_dir=str(tmp_path),
    )
    assert (
        set(loader.genotype)
        == set(loader.sumstats_table)
        == set(loader.annotation)
        == {1, 2, 23}
    )
    np.testing.assert_array_equal(loader.sumstats_table[1].table["POS"], [10, 20])
    np.testing.assert_allclose(loader.sumstats_table[1].table["BETA"], [0.1, -0.2])
    for c in loader.chromosomes:
        assert (
            loader.genotype[c].chromosome
            == loader.sumstats_table[c].chromosome
            == loader.annotation[c].chromosome
            == c
        )


def test_ld_store_chromosome_is_normalized_without_rewriting_source(tmp_path):
    path = str(tmp_path / "legacy_ld")
    ld = LDMatrix.from_csr(csr_matrix((2, 2)), path, dtype="float32")
    ld.set_store_attr("Chromosome", "CHR:1")
    ld.set_metadata("snps", np.array(["rs1", "rs2"]))
    ld.set_metadata("bp", np.array([10, 20]))
    ld.set_metadata("a1", np.array(["A", "C"]))
    ld.set_metadata("a2", np.array(["G", "T"]))
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "CHR\tPOS\tA1\tA2\tBETA\tSE\nchr_1\t10\tA\tG\t.1\t.01\n1\t20\tC\tT\t.2\t.02\n"
    )
    loader = GWADataLoader(
        ld_store_files=path, sumstats_files=str(summary), temp_dir=str(tmp_path)
    )
    assert set(loader.ld) == set(loader.sumstats_table) == {1}
    assert loader.ld[1].chromosome == 1
    assert loader.ld[1].get_store_attr("Chromosome") == "CHR:1"


def test_nonstandard_contigs_and_numeric_chromosomes_can_coexist(tmp_path):
    table = pd.DataFrame(
        {
            "CHR": ["chrUn_KI270442v1", "chr1"],
            "POS": [10, 20],
            "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"],
        }
    )
    summary = SumstatsTable(table.copy())
    annotation = AnnotationMatrix(table.copy())
    genotype = MagenpyGenotypeMatrix(snp_table=table.copy(), temp_dir=str(tmp_path))
    assert summary.chromosomes == [1, "Un_KI270442v1"]
    assert genotype.chromosomes.tolist() == [1, "Un_KI270442v1"]
    assert (
        set(summary.split_by_chromosome())
        == set(annotation.split_by_chromosome())
        == set(genotype.split_by_chromosome())
    )


def test_custom_parsers_also_reach_normalized_constructor_boundaries(tmp_path):
    class SummaryParser:
        def parse(self, *args, **kwargs):
            return pd.DataFrame({"CHR": ["chr:1"], "POS": [10], "A1": ["A"]})

    class AnnotationParser:
        def parse(self, *args, **kwargs):
            return pd.DataFrame({"CHR": ["CHR_1"], "POS": [10], "coding": [1]}), [
                "coding"
            ]

    summary = SumstatsTable.from_file("unused", parser=SummaryParser())
    annotation = AnnotationMatrix.from_file("unused", annot_parser=AnnotationParser())
    assert (
        set(summary.split_by_chromosome())
        == set(annotation.split_by_chromosome())
        == {1}
    )


def test_normalization_precedes_missing_essential_chromosome_filter(tmp_path):
    path = tmp_path / "missing.tsv"
    path.write_text(
        "CHR\tPOS\tA1\tBETA\tSE\nchr:\t10\tA\t.1\t.01\nchr1\t20\tC\t.2\t.02\n"
    )
    table = SumstatsParser(sep="\t").parse(path)
    np.testing.assert_array_equal(table["CHR"], [1])
    np.testing.assert_array_equal(table["POS"], [20])
    retained = SumstatsParser(sep="\t").parse(path, drop_na=False)
    assert pd.isna(retained["CHR"].iloc[0])


def test_xarray_chromosome_coordinates_match_normalized_metadata(tmp_path):
    xr = pytest.importorskip("xarray")
    from magenpy.GenotypeMatrix import xarrayGenotypeMatrix

    matrix = xarrayGenotypeMatrix(
        snp_table=pd.DataFrame({"CHR": ["chr1", "chr:1"], "SNP": ["rs1", "rs2"]}),
        xr_mat=xr.DataArray(
            np.zeros((2, 2)),
            dims=("sample", "variant"),
            coords={"chrom": ("variant", ["chr1", "chr:1"])},
        ),
        temp_dir=str(tmp_path),
    )
    assert matrix.chromosome == 1
    np.testing.assert_array_equal(matrix.xr_mat.coords["chrom"], [1, 1])


def test_standard_chromosome_column_uses_one_byte_per_row():
    n = 100_000
    table = pd.DataFrame({"CHR": np.ones(n, dtype=np.int64)})
    normalize_chromosome_column(table)
    assert table["CHR"].cat.codes.nbytes == n
    assert table["CHR"].memory_usage(index=False, deep=True) < 2 * n


@pytest.mark.parametrize("labels,expected", [([1, 23], {1, 23}), ([None, None], set())])
def test_splits_exclude_unused_plink_categories(tmp_path, labels, expected):
    table = pd.DataFrame(
        {"CHR": labels, "SNP": ["rs1", "rs2"], "POS": [10, 20], "A1": ["A", "C"]}
    )
    objects = [
        SumstatsTable(table.copy()),
        AnnotationMatrix(table.copy()),
        MagenpyGenotypeMatrix(snp_table=table.copy(), temp_dir=str(tmp_path)),
    ]
    for obj in objects:
        assert set(obj.split_by_chromosome()) == expected
        chromosome_column = (
            obj.snp_table["CHR"] if hasattr(obj, "snp_table") else obj.table["CHR"]
        )
        assert len(chromosome_column.cat.categories) == 27


def test_harmonization_compares_contig_labels_across_different_category_codes():
    from magenpy.utils.model_utils import build_snp_harmonization_plan

    reference = pd.DataFrame({"CHR": ["chrZZ"], "POS": [10]})
    other = pd.DataFrame({"CHR": ["chrAA", "ZZ"], "POS": [5, 10]})
    normalize_chromosome_column(reference)
    normalize_chromosome_column(other)
    assert reference["CHR"].cat.codes.iloc[0] == 27
    assert other["CHR"].cat.codes.iloc[1] == 28
    plan = build_snp_harmonization_plan(
        {
            "reference": pd.MultiIndex.from_frame(reference),
            "other": pd.MultiIndex.from_frame(other),
        },
        reference="reference",
    )
    np.testing.assert_array_equal(plan["indexers"]["reference"], [0])
    np.testing.assert_array_equal(plan["indexers"]["other"], [1])
    original = other["CHR"].array
    normalize_chromosome_column(other)
    assert other["CHR"].array is original


def test_category_codes_widen_when_many_nonstandard_contigs_are_present():
    labels = [f"chrContig{i}" for i in range(150)]
    table = pd.DataFrame({"CHR": labels})
    normalize_chromosome_column(table)
    assert table["CHR"].cat.codes.dtype == np.int16
    assert table["CHR"].tolist() == [f"Contig{i}" for i in range(150)]


@pytest.mark.parametrize(
    "constructor,argument",
    [
        (AnnotationMatrix, "annotation_table"),
        (SumstatsTable, "ss_table"),
        (MagenpyGenotypeMatrix, "snp_table"),
    ],
)
@pytest.mark.parametrize("labels", [["chr1", "chr2"], ["chr1", "chrZZ"]])
def test_construction_and_splitting_skip_normalization_for_parsed_columns(
    tmp_path, monkeypatch, constructor, argument, labels
):
    from importlib import import_module

    table = pd.DataFrame(
        {"CHR": labels, "SNP": ["rs1", "rs2"], "POS": [10, 20], "A1": ["A", "C"]}
    )
    normalize_chromosome_column(table)
    original = table["CHR"].array

    def redundant_normalization(*args, **kwargs):
        pytest.fail("Already-parsed chromosome columns must bypass normalization.")

    monkeypatch.setattr(
        import_module(constructor.__module__),
        "normalize_chromosome_column",
        redundant_normalization,
    )
    kwargs = {argument: table}
    if constructor is MagenpyGenotypeMatrix:
        kwargs["temp_dir"] = str(tmp_path)
    data = constructor(**kwargs)
    assert table["CHR"].array is original
    assert set(data.split_by_chromosome()) == set(table["CHR"].unique())
