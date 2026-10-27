import numpy as np
import pandas as pd
import pytest

from magenpy.SumstatsTable import SumstatsTable
from magenpy.GWADataLoader import GWADataLoader
from magenpy.parsers.sumstats_parsers import (
    COJOSSParser,
    FastGWASSParser,
    LDAKSSParser,
    Plink2SSParser,
    RegenieSSParser,
    SaigeSSParser,
    SumstatsParser,
    get_sumstats_parser,
    get_supported_sumstats_formats,
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


def test_known_sumstats_columns_use_explicit_dtypes(tmp_path):
    sumstats_file = tmp_path / 'fastgwa.tsv'
    pd.DataFrame({
        'CHR': [22],
        'SNP': ['rs1'],
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

    assert pd.api.types.is_string_dtype(parsed['SNP'])
    assert pd.api.types.is_string_dtype(parsed['A1'])
    assert parsed['POS'].dtype == np.dtype(np.int32)
    assert parsed['N'].dtype == np.dtype(np.float64)
    for column in ('MAF', 'BETA', 'SE', 'PVAL'):
        assert parsed[column].dtype == np.dtype(np.float64)


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


def test_regenie_parser_maps_effect_allele_and_log_pvalue(tmp_path):
    sumstats_file = tmp_path / 'trait.regenie'
    sumstats_file.write_text(
        'CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA\n'
        '1 12345 rs1 G A 0.2 0.99 1000 ADD 0.1 0.01 100 12.5 NA\n'
    )

    parsed = RegenieSSParser().parse(sumstats_file)

    assert parsed.loc[0, 'CHR'] == 1
    assert parsed.loc[0, 'POS'] == 12345
    assert parsed.loc[0, 'SNP'] == 'rs1'
    assert parsed.loc[0, 'A1'] == 'A'
    assert parsed.loc[0, 'A2'] == 'G'
    assert parsed.loc[0, 'MAF'] == 0.2
    assert parsed.loc[0, 'PVAL'] == pytest.approx(10 ** -12.5)


@pytest.mark.parametrize(
    ('stat_columns', 'expected_z'),
    [
        ({'Z': -2.5}, -2.5),
        ({'BETA': -0.5, 'SE': 0.2}, None),
        ({'Stat': 6.25, 'Direction': -1}, -2.5),
        ({'P': 0.012419330651552265, 'Direction': -1}, -2.5),
    ],
)
def test_ldak_parser_supports_all_statistic_layouts(
    tmp_path, stat_columns, expected_z
):
    sumstats_file = tmp_path / 'trait.summaries'
    row = {
        'Predictor': 'rs1',
        'A1': 'A',
        'A2': 'G',
        'n': 1000,
        'A1Freq': 0.2,
        **stat_columns,
    }
    pd.DataFrame([row]).to_csv(sumstats_file, sep=' ', index=False)

    parsed = LDAKSSParser().parse(sumstats_file)

    assert parsed.loc[0, 'SNP'] == 'rs1'
    assert parsed.loc[0, 'N'] == 1000
    assert parsed.loc[0, 'MAF'] == 0.2
    if expected_z is None:
        assert parsed.loc[0, 'BETA'] == -0.5
        assert parsed.loc[0, 'SE'] == 0.2
    else:
        assert parsed.loc[0, 'Z'] == pytest.approx(expected_z)


def test_ldak_parser_supports_assoc_output(tmp_path):
    sumstats_file = tmp_path / 'trait.assoc'
    sumstats_file.write_text(
        'Chromosome Predictor Basepair A1 A2 Wald_Stat Wald_P Effect SD A1_mean MAF\n'
        '1 rs1 12345 A G -2.5 0.0124 -0.5 0.2 0.4 0.2\n'
    )

    parsed = LDAKSSParser().parse(sumstats_file)

    assert parsed.loc[0, 'CHR'] == 1
    assert parsed.loc[0, 'POS'] == 12345
    assert parsed.loc[0, 'SNP'] == 'rs1'
    assert parsed.loc[0, 'Z'] == -2.5
    assert parsed.loc[0, 'PVAL'] == 0.0124
    assert parsed.loc[0, 'BETA'] == -0.5
    assert parsed.loc[0, 'SE'] == 0.2
    assert parsed.loc[0, 'MAF'] == 0.2


def test_registered_formats_propagate_to_public_entry_points():
    canonical = get_supported_sumstats_formats()
    aliases = get_supported_sumstats_formats(include_aliases=True)

    assert {'regenie', 'ldak'} <= set(canonical)
    assert {'regenie', 'ldak', 'sumher', 'megaprs'} <= set(aliases)
    assert get_sumstats_parser('REGENIE') is RegenieSSParser
    assert get_sumstats_parser('SumHer') is LDAKSSParser
    assert SumstatsTable.supported_sumstats_formats() == canonical
    assert GWADataLoader.supported_sumstats_formats(True) == aliases
