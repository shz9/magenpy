import numpy as np
import magenpy as mgp
import shutil
import pytest


@pytest.fixture(scope='module')
def gdl_object():
    """
    Initialize a GWADataLoader using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            sumstats_files=mgp.ukb_height_sumstats_path(),
                            sumstats_format='fastgwa',
                            backend='xarray')

    yield gdl

    # Clean up after tests are done:
    gdl.cleanup()
    shutil.rmtree(gdl.temp_dir)
    shutil.rmtree(gdl.output_dir)


def test_basic_properties(gdl_object):
    """
    Test the basic properties of the GWADataLoader object.
    """

    # Check basic shapes parameters:
    assert gdl_object is not None
    assert len(gdl_object.sample_table) == gdl_object.n == 378  # Sample size
    assert gdl_object.m == len(gdl_object.sumstats_table[22]) == 15935  # Number of variants
    assert gdl_object.shapes == {22: 15935}  # Number of variants per chromosome
    assert gdl_object.chromosomes == [22]  # List of chromosomes
    assert gdl_object.n_annotations is None  # Number of annotations

    # Check that the individual data sources have been properly harmonized:

    assert gdl_object.genotype is not None
    assert gdl_object.sumstats_table is not None

    assert gdl_object.genotype[22].n_snps == gdl_object.n_snps == gdl_object.sumstats_table[22].n_snps
    assert gdl_object.genotype[22].n == gdl_object.n
    # Check that the variant IDs are harmonized:
    assert np.array_equal(gdl_object.genotype[22].snps, gdl_object.sumstats_table[22].snps)
    # Check that the variant positions are harmonized:
    assert np.array_equal(gdl_object.genotype[22].bp_pos, gdl_object.sumstats_table[22].bp_pos)
    # Check that the alternative alleles are harmonized:
    assert np.array_equal(gdl_object.genotype[22].a1, gdl_object.sumstats_table[22].a1)


def test_filtering_methods(gdl_object):
    """
    Test the filtering methods of the GWADataLoader object. Primarily,
    we test the `filter_samples` and `filter_snps` methods to make sure
    they are behaving as expected.
    """

    # Filter the samples:

    # First draw a random subset of samples to keep:
    np.random.seed(0)
    keep_samples = np.random.choice(gdl_object.samples, size=100, replace=False)
    # Then apply the filtering method:
    gdl_object.filter_samples(keep_samples=keep_samples)

    assert gdl_object.n == gdl_object.genotype[22].n == 100

    # Filter the SNPs:

    # First draw a random subset of SNPs to keep:
    keep_snps = np.random.choice(gdl_object.snps[22], size=3000, replace=False)
    gdl_object.filter_snps(extract_snps=keep_snps, chromosome=22)

    assert gdl_object.n_snps == gdl_object.genotype[22].n_snps == 3000
