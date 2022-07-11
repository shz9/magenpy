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


class LDSCRegression(object):

    def __init__(self, gdl: mgp.GWADataLoader, n_blocks=200, max_chisq=None):
        """
        Incomplete...
        """

        self.gdl = gdl
        self.n_blocks = n_blocks

        # Extract the data from the GDL object:

        chroms = self.gdl.chromosomes

        if self.gdl.annotation is not None:
            self.ld_scores = np.concatenate([
                self.gdl.ld[c].compute_ld_scores(
                    annotations=self.gdl.annotation[c].values(add_intercept=True)
                )
                for c in chroms
            ])
        else:
            self.ld_scores = np.concatenate([self.gdl.ld[c].ld_score.reshape(-1, 1) for c in chroms])

        self.chisq = np.concatenate([self.gdl.sumstats_table[c].get_chisq_statistic() for c in chroms])
        self.n = np.concatenate([self.gdl.sumstats_table[c].n_per_snp for c in chroms])

        if max_chisq is None:
            max_chisq = max(0.001*self.n.max(), 80)

        chisq_cond = self.chisq < max_chisq

        self.ld_scores = self.ld_scores[chisq_cond, :]
        self.chisq = self.chisq[chisq_cond]
        self.n = self.n[chisq_cond]

    def fit(self):
        """
        TODO: Implement the jackknife estimator here...
        """
        pass

