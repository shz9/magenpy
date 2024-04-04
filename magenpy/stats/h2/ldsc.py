import numpy as np
from ...GWADataLoader import GWADataLoader


def simple_ldsc(gdl: GWADataLoader):
    """
    Provides an estimate of SNP heritability from summary statistics using
    a simplified version of the LD Score Regression framework.
    E[X_j^2] = h^2*l_j + int
    Where the response is the Chi-Squared statistic for SNP j
    and the variable is its LD score.

    :param gdl: An instance of `GWADataLoader` with the LD information and
    summary statistics initialized properly.

    :return: The estimated SNP heritability.
    """

    # Check data types:
    assert gdl.ld is not None and gdl.sumstats_table is not None

    ld_score = []
    chi_sq = []
    sample_size = []

    for c in gdl.chromosomes:
        ld_score.append(gdl.ld[c].ld_score)
        chi_sq.append(gdl.sumstats_table[c].get_chisq_statistic())
        sample_size.append(gdl.sumstats_table[c].n_per_snp.max())

    ld_score = np.concatenate(ld_score)
    chi_sq = np.concatenate(chi_sq)
    sample_size = max(sample_size)

    return (chi_sq.mean() - 1.) * len(ld_score) / (ld_score.mean() * sample_size)


class LDSCRegression(object):
    """
    Perform LD Score Regression using the jackknife method.
    """

    def __init__(self, gdl: GWADataLoader, n_blocks=200, max_chisq=None):
        """
        :param gdl: An instance of GWADataLoader
        :param n_blocks: The number of blocks to use for the jackknife method.
        :param max_chisq: The maximum Chi-Squared statistic to consider.
        """

        self.gdl = gdl
        self.n_blocks = n_blocks

        # ...

    def fit(self):
        """
        Perform LD Score Regression estimation using the jackknife method.

        :raises NotImplementedError: If method is not implemented.
        """

        raise NotImplementedError

