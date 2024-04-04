import magenpy as mgp
import numpy as np
import shutil
import pytest


@pytest.fixture(scope='module')
def gsim_object():
    """
    Initialize a GWADataLoader using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    gsim = mgp.PhenotypeSimulator(mgp.tgp_eur_data_path(),
                                  backend='xarray')

    yield gsim

    # Clean up after tests are done:
    gsim.cleanup()
    shutil.rmtree(gsim.temp_dir)
    shutil.rmtree(gsim.output_dir)


def test_simulator(gsim_object):
    """
    Test the basic functionality of the phenotype simulator
    """

    gsim_object.simulate()

    assert gsim_object.sample_table is not None
    assert gsim_object.sample_table.phenotype is not None
    assert gsim_object.sample_table.phenotype_likelihood == 'gaussian'
    assert len(gsim_object.sample_table.phenotype) == gsim_object.sample_size

    gsim_object.phenotype_likelihood = 'binomial'

    gsim_object.simulate()

    assert gsim_object.sample_table is not None
    assert gsim_object.sample_table.phenotype is not None
    assert gsim_object.sample_table.phenotype_likelihood == 'binomial'
    assert sorted(np.unique(gsim_object.sample_table.phenotype)) == [0, 1]

