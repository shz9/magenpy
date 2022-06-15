import numpy as np
import magenpy as mgp


def simple_ldsc(gdl: mgp.GWADataLoader):
    """
    Provides an estimate of SNP heritability from summary statistics using
    a simplified version of the LD Score Regression framework.
    E[X_j^2] = h^2*l_j + int
    Where the response is the Chi-Squared statistic for SNP j
    and the variable is its LD score.

    NOTE: For now, we constrain the slope to 1.

    :param gdl: An instance of `GWADataLoader` with the LD information and
    summary statistics initialized properly.
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
