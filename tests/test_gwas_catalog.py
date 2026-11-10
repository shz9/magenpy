import pandas as pd

from magenpy.GWADataLoader import GWADataLoader
from magenpy.SumstatsTable import SumstatsTable
from magenpy.utils.gwas_catalog import (
    gwas_catalog_directory,
    is_gwas_catalog_accession,
    resolve_gwas_catalog,
)


def test_gwas_catalog_accession_and_directory():
    assert is_gwas_catalog_accession(" gcst005195 ")
    assert not is_gwas_catalog_accession("PGS005195")
    assert gwas_catalog_directory("GCST005195") == (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST005001-GCST006000/GCST005195/"
    )


def test_resolve_gwas_catalog_prefers_harmonised_build_37():
    directory = gwas_catalog_directory("GCST005195")
    harmonised = directory + "harmonised/"
    formatted = harmonised + "study_buildGRCh37.f.tsv.gz"
    preferred = harmonised + "study_buildGRCh37.h.tsv.gz"
    listings = {
        directory: f'<a href="harmonised/">harmonised</a>',
        harmonised: (
            f'<a href="{formatted}">formatted</a>'
            f'<a href="{preferred}">harmonised</a>'
        ),
    }

    assert resolve_gwas_catalog("GCST005195", listings.__getitem__) == preferred


def test_resolve_gwas_catalog_uses_metadata_for_build():
    directory = gwas_catalog_directory("GCST005195")
    sumstats_url = directory + "study.tsv.gz"
    metadata_url = sumstats_url + "-meta.yaml"
    listings = {
        directory: (
            '<a href="study.tsv.gz">sumstats</a>'
            '<a href="study.tsv.gz-meta.yaml">metadata</a>'
        ),
        metadata_url: "genome_assembly: GRCh37\n",
    }

    assert resolve_gwas_catalog("GCST005195", listings.__getitem__) == sumstats_url


def test_sumstats_table_loads_gwas_catalog_accession(monkeypatch, tmp_path):
    sumstats_file = tmp_path / "catalog.tsv"
    pd.DataFrame(
        {
            "chromosome": [1],
            "base_pair_location": [123],
            "rsid": ["rs1"],
            "effect_allele": ["A"],
            "other_allele": ["G"],
            "beta": [0.1],
            "standard_error": [0.01],
        }
    ).to_csv(sumstats_file, sep="\t", index=False)
    monkeypatch.setattr(
        "magenpy.utils.gwas_catalog.resolve_gwas_catalog",
        lambda accession: sumstats_file,
    )

    sumstats = SumstatsTable.from_file("GCST005195")

    assert sumstats.table.loc[0, "SNP"] == "rs1"
    assert sumstats.table.loc[0, "BETA"] == 0.1
    assert sumstats.genome_build == "GRCh37"


def test_gwa_data_loader_accepts_gwas_catalog_accession(monkeypatch, tmp_path):
    sumstats_file = tmp_path / "catalog.tsv"
    pd.DataFrame(
        {
            "chromosome": [1],
            "base_pair_location": [123],
            "rsid": ["rs1"],
            "effect_allele": ["A"],
            "other_allele": ["G"],
            "beta": [0.1],
            "standard_error": [0.01],
        }
    ).to_csv(sumstats_file, sep="\t", index=False)
    monkeypatch.setattr(
        "magenpy.utils.gwas_catalog.resolve_gwas_catalog",
        lambda accession: sumstats_file,
    )
    loader = GWADataLoader.__new__(GWADataLoader)
    loader.genotype = None
    loader.ld = None

    loader.read_summary_statistics("GCST005195")

    assert loader.sumstats_table[1].table.loc[0, "SNP"] == "rs1"
