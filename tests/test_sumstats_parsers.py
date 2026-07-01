import numpy as np
import pandas as pd

from magenpy.SumstatsTable import SumstatsTable
from magenpy.parsers.sumstats_parsers import COJOSSParser, SumstatsParser


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
