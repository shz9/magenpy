import numpy as np
import magenpy as mgp
import shutil
import pytest


def _basic_ld_checks(gdl: mgp.GWADataLoader):
    """
    Perform basic checks on the computed  LD matrix, such as its dimensions,
    sample size, and the number of SNPs.

    :param gdl: An instance of `GWADataLoader`

    :return: True if all checks pass, raises AssertionError otherwise.
    """

    assert gdl.ld is not None
    assert gdl.ld[22] is not None
    assert gdl.ld[22].validate_ld_matrix()
    assert gdl.ld[22].stored_n_snps == gdl.n_snps
    assert gdl.ld[22].sample_size == gdl.sample_size

    assert np.array_equal(gdl.ld[22].snps, gdl.snps[22])
    # Add other checks?
    # ...

    return True


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


@pytest.fixture(scope='module')
def sample_ld(gdl_object: mgp.GWADataLoader):
    """
    Test the LD computation functionality according to the Sample estimator
    """

    gdl_object.compute_ld('sample', gdl_object.output_dir)
    gdl_object.harmonize_data()

    _basic_ld_checks(gdl_object)

    yield gdl_object.ld[22]


@pytest.fixture(scope='module')
def windowed_ld(gdl_object: mgp.GWADataLoader):
    """
    Test the LD computation functionality according to the Windowed estimator
    """

    gdl_object.compute_ld('windowed',
                          gdl_object.output_dir,
                          compute_spectral_properties=True,
                          window_size=500,
                          kb_window_size=100,
                          cm_window_size=3.)
    gdl_object.harmonize_data()

    _basic_ld_checks(gdl_object)

    yield gdl_object.ld[22]


@pytest.fixture(scope='module')
def shrinkage_ld(gdl_object: mgp.GWADataLoader):
    """
    Test the LD computation functionality according to the Shrinkage estimator
    """

    gdl_object.compute_ld('shrinkage',
                          gdl_object.output_dir,
                          genetic_map_ne=11400,
                          genetic_map_sample_size=183)
    gdl_object.harmonize_data()

    _basic_ld_checks(gdl_object)

    yield gdl_object.ld[22]


@pytest.fixture(scope='module')
def block_ld(gdl_object: mgp.GWADataLoader):
    """
    Test the LD computation functionality according to the Block estimator
    """

    ld_block_url = ("https://bitbucket.org/nygcresearch/ldetect-data/raw/"
                    "ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed")
    gdl_object.compute_ld('block', gdl_object.output_dir, ld_blocks_file=ld_block_url)
    gdl_object.harmonize_data()

    _basic_ld_checks(gdl_object)

    yield gdl_object.ld[22]


def test_basic_ld_filtering(block_ld: mgp.LDMatrix):
    """
    Test the LD filtering functionality
    """

    # First release the data from memory and reset the mask:
    block_ld.reset_mask()
    block_ld.release()

    orig_n_snps = block_ld.n_snps

    # Sample a subset of SNPs:
    snp_subset = np.random.choice(block_ld.snps, 100, replace=False)
    block_ld.filter_snps(snp_subset)

    # Check that the number of SNPs is as expected:
    assert block_ld.n_snps == len(snp_subset)
    assert block_ld.shape == (len(snp_subset), len(snp_subset))

    intersected_snps = np.intersect1d(block_ld.snps, snp_subset)

    assert len(intersected_snps) == len(snp_subset)
    assert np.all(np.isin(snp_subset, block_ld.snps))

    # Check the snp table:
    assert block_ld.to_snp_table().shape[0] == len(snp_subset)

    # Load a linear operator object:
    lop = block_ld.to_linear_operator()

    assert lop.shape == block_ld.shape

    # check multiplication:
    x = np.random.rand(block_ld.n_snps)
    assert block_ld.dot(x).shape == x.shape

    # Check that attributes regarding the original data are still correct:
    assert block_ld.stored_n_snps == orig_n_snps
    assert block_ld.stored_shape == (orig_n_snps, orig_n_snps)


def test_ld_linear_operator(block_ld: mgp.LDMatrix):
    """
    Test the Linear Operator functionality
    """

    # First release the data from memory and reset the mask:
    block_ld.reset_mask()
    block_ld.release()

    # Load a linear operator object:
    lop = block_ld.to_linear_operator()

    assert lop.shape == block_ld.shape

    # check multiplication:
    x = np.random.rand(block_ld.n_snps)
    lop_dot = lop.dot(x)
    assert lop_dot.shape == x.shape

    # Check the data type:
    assert block_ld.dtype == lop.ld_data_type

    # Check CSR functionality:
    csr_mat = lop.to_csr()
    assert csr_mat.shape == lop.shape

    # Check that multiplication with CSR yields the same result:
    csr_dot = csr_mat.dot(x)
    assert np.allclose(csr_dot, lop_dot)

    # Set up different boundaries for testing slicing functionality:
    boundaries = [(0, 100), (150, 250), (lop.shape[0] - 100, lop.shape[0])]

    for start, end in boundaries:

        # Extract a numpy block from the linear operator directly:
        numpy_block = lop.to_numpy(block_start=start, block_end=end)

        # Extract the same block from the CSR matrix:
        csr_block = csr_mat[start:end, start:end].toarray()

        # Slice the LOP object:
        lop_block = lop[start:end, start:end]

        # Check that all the resulting shapes are compatible:
        assert numpy_block.shape == csr_block.shape == lop_block.shape

        # Load the lop block into a numpy array:
        lop_block = lop_block.to_numpy()

        assert np.allclose(lop_block, csr_block) and np.allclose(numpy_block, csr_block)

        # Check that the block is symmetric:
        assert np.allclose(numpy_block, numpy_block.T)

        # Check that the block contains all ones along the diagonal:
        assert np.allclose(np.diag(numpy_block), 1.)

