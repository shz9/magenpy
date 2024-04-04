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
                            backend='xarray')

    # Extract a smaller subset of variants for testing:
    np.random.seed(0)
    keep_snps = gdl.snps[22][np.random.choice(2000, 1000, replace=False)]
    gdl.filter_snps(extract_snps=keep_snps, chromosome=22)

    yield gdl

    # Clean up after tests are done:
    gdl.cleanup()
    shutil.rmtree(gdl.temp_dir)
    shutil.rmtree(gdl.output_dir)


def test_sample_ld_computation(gdl_object):
    """
    Test the LD computation functionality according to the Sample estimator
    """

    gdl_object.compute_ld('sample', gdl_object.output_dir)
    gdl_object.harmonize_data()

    # Check that the LD matrix has been computed:
    assert gdl_object.ld is not None
    assert gdl_object.ld[22] is not None
    assert gdl_object.ld[22].validate_ld_matrix()
    assert gdl_object.ld[22].stored_n_snps == gdl_object.n_snps
    assert gdl_object.ld[22].sample_size == gdl_object.sample_size

    assert np.array_equal(gdl_object.ld[22].snps, gdl_object.snps[22])

    # Add other checks?


def test_windowed_ld_computation(gdl_object):
    """
    Test the LD computation functionality according to the Windowed estimator
    """

    gdl_object.compute_ld('windowed',
                          gdl_object.output_dir,
                          window_size=500,
                          kb_window_size=100,
                          cm_window_size=3.)
    gdl_object.harmonize_data()

    # Check that the LD matrix has been computed:
    assert gdl_object.ld is not None
    assert gdl_object.ld[22] is not None
    assert gdl_object.ld[22].validate_ld_matrix()
    assert gdl_object.ld[22].stored_n_snps == gdl_object.n_snps
    assert gdl_object.ld[22].sample_size == gdl_object.sample_size

    assert np.array_equal(gdl_object.ld[22].snps, gdl_object.snps[22])

    # Add other checks?


def test_shrinkage_ld_computation(gdl_object):
    """
    Test the LD computation functionality according to the Shrinkage estimator
    """

    gdl_object.compute_ld('shrinkage',
                          gdl_object.output_dir,
                          genetic_map_ne=11400,
                          genetic_map_sample_size=183)
    gdl_object.harmonize_data()

    # Check that the LD matrix has been computed:
    assert gdl_object.ld is not None
    assert gdl_object.ld[22] is not None
    assert gdl_object.ld[22].validate_ld_matrix()
    assert gdl_object.ld[22].stored_n_snps == gdl_object.n_snps
    assert gdl_object.ld[22].sample_size == gdl_object.sample_size

    assert np.array_equal(gdl_object.ld[22].snps, gdl_object.snps[22])

    # Add other checks?


def test_block_ld_computation(gdl_object):
    """
    Test the LD computation functionality according to the Block estimator
    """

    ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
    gdl_object.compute_ld('block', gdl_object.output_dir, ld_blocks_file=ld_block_url)
    gdl_object.harmonize_data()

    # Check that the LD matrix has been computed:
    assert gdl_object.ld is not None
    assert gdl_object.ld[22] is not None
    assert gdl_object.ld[22].validate_ld_matrix()
    assert gdl_object.ld[22].stored_n_snps == gdl_object.n_snps
    assert gdl_object.ld[22].sample_size == gdl_object.sample_size

    assert np.array_equal(gdl_object.ld[22].snps, gdl_object.snps[22])

    # Add other checks?

