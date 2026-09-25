import numpy as np
import pandas as pd

from magenpy.SumstatsTable import SumstatsTable
from magenpy.parsers.sumstats_parsers import (
    COJOSSParser,
    FastGWASSParser,
    Plink2SSParser,
    SaigeSSParser,
    SumstatsParser,
)


def test_essential_cols_resolve_to_first_available_column_sets():
    columns = ['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA', 'SE', 'Z']

    assert SumstatsParser.get_essential_cols(columns) == [
        'SNP', 'A1', 'BETA', 'SE'
    ]


def test_essential_cols_fall_back_to_position_when_snp_is_missing():
    columns = ['CHR', 'POS', 'A1', 'A2', 'Z']

    assert SumstatsParser.get_essential_cols(columns) == [
        'CHR', 'POS', 'A1', 'Z'
    ]


def test_drop_na_is_restricted_to_resolved_essential_columns(tmp_path):
    sumstats_file = tmp_path / 'sumstats.tsv'
    pd.DataFrame({
        'CHR': [1, 1, 1],
        'SNP': ['rs1', 'rs2', 'rs3'],
        'A1': ['A', 'C', 'G'],
        'A2': ['G', 'T', np.nan],
        'BETA': [0.1, np.nan, -0.2],
        'SE': [0.01, 0.02, 0.03],
        'MAF': [np.nan, 0.2, 0.3],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = SumstatsParser(sep='\t').parse(sumstats_file)

    assert parsed['SNP'].tolist() == ['rs1', 'rs3']
    assert parsed.loc[parsed['SNP'] == 'rs1', 'MAF'].isna().all()
    assert parsed.loc[parsed['SNP'] == 'rs3', 'A2'].isna().all()


def test_to_file_uses_parser_standard_cols_and_output_names(tmp_path):
    output_file = tmp_path / 'sumstats.ma'
    sumstats = SumstatsTable(pd.DataFrame({
        'CHR': [1],
        'SNP': ['rs1'],
        'A1': ['A'],
        'A2': ['G'],
        'MAF': [0.2],
        'BETA': [0.1],
        'SE': [0.01],
        'PVAL': [0.05],
        'N': [1000],
    }))

    sumstats.to_file(output_file, sumstats_format='cojo')
    written = pd.read_csv(output_file, sep='\t')

    assert written.columns.tolist() == [
        'SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'N'
    ]


def test_cojo_parser_exposes_standard_output_columns():
    assert COJOSSParser.get_standard_cols() == [
        'SNP', 'A1', 'A2', 'MAF', 'BETA', 'SE', 'PVAL', 'N'
    ]


def test_identifier_dtypes_preserve_leading_zeroes(tmp_path):
    sumstats_file = tmp_path / 'fastgwa.tsv'
    pd.DataFrame({
        'CHR': [22],
        'SNP': ['00123'],
        'POS': [12345],
        'A1': ['A'],
        'A2': ['G'],
        'N': [1000],
        'AF1': [0.2],
        'BETA': [0.1],
        'SE': [0.01],
        'P': [0.05],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = FastGWASSParser(sep='\t').parse(sumstats_file)

    assert parsed.loc[0, 'SNP'] == '00123'
    assert parsed['POS'].dtype == np.dtype(np.int32)


def test_missing_nonessential_position_uses_nullable_int32(tmp_path):
    sumstats_file = tmp_path / 'sumstats.tsv'
    pd.DataFrame({
        'SNP': ['rs1', 'rs2'],
        'POS': [100, np.nan],
        'A1': ['A', 'C'],
        'BETA': [0.1, 0.2],
        'SE': [0.01, 0.02],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = SumstatsParser(sep='\t').parse(sumstats_file)

    assert parsed['SNP'].tolist() == ['rs1', 'rs2']
    assert str(parsed['POS'].dtype) == 'Int32'
    assert pd.isna(parsed.loc[1, 'POS'])


def test_user_dtype_overrides_parser_default(tmp_path):
    sumstats_file = tmp_path / 'sumstats.tsv'
    pd.DataFrame({
        'SNP': ['rs1'],
        'A1': ['A'],
        'BETA': [0.1],
        'SE': [0.01],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = SumstatsParser(sep='\t', dtype={'SNP': 'category'}).parse(sumstats_file)

    assert isinstance(parsed['SNP'].dtype, pd.CategoricalDtype)


def test_plink2_postprocessing_precedes_missing_value_filter(tmp_path):
    sumstats_file = tmp_path / 'plink2.tsv'
    pd.DataFrame({
        '#CHROM': [1, 1],
        'POS': [100, 200],
        'ID': ['rs1', 'rs2'],
        'REF': ['G', 'T'],
        'ALT': ['A', 'C'],
        'A1': ['A', np.nan],
        'BETA': [0.1, 0.2],
        'SE': [0.01, 0.02],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = Plink2SSParser(sep='\t').parse(sumstats_file)

    assert parsed['SNP'].tolist() == ['rs1']
    assert parsed['A2'].tolist() == ['G']


def test_saige_sample_size_is_inferred_in_single_pipeline(tmp_path):
    sumstats_file = tmp_path / 'saige.tsv'
    pd.DataFrame({
        'MarkerID': ['rs1'],
        'Allele1': ['G'],
        'Allele2': ['A'],
        'AF_Allele2': [0.25],
        'AC_Allele2': [500],
        'Tstat': [2.0],
        'p.value': [0.05],
    }).to_csv(sumstats_file, sep='\t', index=False)

    parsed = SaigeSSParser(sep='\t').parse(sumstats_file)

    assert parsed.loc[0, 'N'] == 1000


def test_single_chromosome_split_skips_grouping_but_preserves_copy_semantics():
    sumstats = SumstatsTable(pd.DataFrame({
        'CHR': [22, 22],
        'SNP': ['rs1', 'rs2'],
        'A1': ['A', 'C'],
    }))

    split = sumstats.split_by_chromosome()

    assert list(split) == [22]
    assert split[22] is not sumstats
    pd.testing.assert_frame_equal(split[22].table, sumstats.table)
