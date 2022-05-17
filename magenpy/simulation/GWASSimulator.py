"""
Author: Shadi Zabad
Date: March 2021
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from magenpy.utils.model_utils import multinomial_rvs
from magenpy import GWASDataLoader


class GWASSimulator(GWASDataLoader):

    def __init__(self, bed_files,
                 h2g=0.2,
                 pi=(0.9, 0.1),
                 d=(0., 1.),
                 prevalence=0.15,
                 **kwargs):
        """
        Simulate phenotypes using the linear additive model.

        :param bed_files: A path (or list of paths) to PLINK bed files.
        :param h2g: The trait SNP heritability, or proportion of variance explained by SNPs.
        :param pi: The mixture proportions for Gaussian mixture density.
        :param d:  The variance multipliers for each component of the mixture density.
        :param prevalence: The (disease) prevalence for binary (case-control) phenotypes.
        """

        super().__init__(bed_files, **kwargs)

        self.pi = pi
        self.h2g = h2g
        self.prevalence = prevalence

        # Sanity checks:
        assert 0. <= self.h2g <= 1.
        assert round(sum(self.pi), 1) == 1.
        assert 0. < self.prevalence < 1.

        self.d = np.array(d)

        self.per_snp_h2g = None
        self.per_snp_pi = None
        self.betas = None
        self.mixture_assignment = None

    @property
    def n_mixtures(self):
        return len(self.pi)

    def set_pi(self, new_pi):
        """
        Set the mixture proportions (proportion of variants in each
        Gaussian mixture component).
        """
        self.pi = new_pi
        self.set_per_snp_mixture_probability()

    def set_h2g(self, new_h2g):
        """
        Set the total heritability (proportion of additive variance due to SNPs) for the trait
        """
        self.h2g = new_h2g
        self.set_per_snp_heritability()

    def set_per_snp_mixture_probability(self):
        """
        Set the per-SNP mixture probability for each variant in the dataset.
        """

        self.per_snp_pi = {}

        for c, c_size in self.shapes.items():
            self.per_snp_pi[c] = np.repeat(np.array([self.pi]), c_size, axis=0)

    def set_per_snp_heritability(self):
        """
        Set the per-SNP heritability for each variant in the dataset.
        """

        assert self.mixture_assignment is not None

        # Estimate the global sigma_beta_sq based on the
        # pre-specified heritability, the mixture proportions `pi`,
        # and the prior multipliers `d`.
        sigma_beta_sq = self.h2g / (self.M*np.array(self.pi)*self.d).sum()

        self.per_snp_h2g = {}

        for c, c_size in self.shapes.items():
            self.per_snp_h2g[c] = sigma_beta_sq*self.d[np.where(self.mixture_assignment[c])[1]]

    def get_causal_status(self):
        """
        This method returns a dictionary of binary vectors
        indicating which snps are causal for each chromosome
        """

        assert self.mixture_assignment is not None

        try:
            zero_index = list(self.d).index(0)
        except ValueError:
            # If all SNPs are causal:
            return {c: np.repeat(True, c_size) for c, c_size in self.shapes.items()}

        causal_status = {}

        for c, mix_a in self.mixture_assignment.items():
            causal_status[c] = np.where(mix_a)[1] != zero_index

        return causal_status

    def set_mixture_assignment(self, new_assignment):
        """
        Set the mixture assignments according to user-provided dictionary.
        :param new_assignment: A dictionary where the keys are the chromosomes and
        the values are the mixture assignment for each SNP on that chromosome.
        """

        # Check that the shapes match pre-specified information:
        for c, c_size in self.shapes.items():
            assert new_assignment[c].shape == (c_size, self.n_mixtures)

        self.mixture_assignment = new_assignment

    def simulate_mixture_assignment(self):
        """
        Simulate assigning SNPs to the various mixture components
        with probabilities given by `pi`.
        """

        if self.per_snp_pi is None or len(self.per_snp_pi) < 1:
            self.set_per_snp_mixture_probability()

        self.mixture_assignment = {}

        for c, c_size in self.shapes.items():

            self.mixture_assignment[c] = multinomial_rvs(1, self.per_snp_pi[c])

        return self.mixture_assignment

    def set_betas(self, new_betas):
        """
        Set the betas according to user-provided dictionary.
        :param new_betas: A dictionary where the keys are the chromosomes and
        the values are the betas for each SNP on that chromosome.
        """

        # Check that the shapes match pre-specified information:
        for c, c_size in self.shapes.items():
            assert len(new_betas[c]) == c_size

        self.betas = new_betas

    def simulate_betas(self):
        """
        Simulate the causal effect size for the variants included
        in the dataset.
        """

        if self.per_snp_h2g is None or len(self.per_snp_h2g) < 1:
            self.set_per_snp_heritability()

        self.betas = {}

        for c, c_size in self.shapes.items():

            self.betas[c] = np.random.normal(loc=0.0,
                                             scale=np.sqrt(self.per_snp_h2g[c]),
                                             size=c_size)

        return self.betas

    def simulate_phenotypes(self):
        """
        Simulate the phenotypes for N individuals
        """

        assert self.betas is not None

        # Compute the polygenic score given the simulated/provided betas:
        if self.use_plink:
            pgs = self.score_plink(self.betas)
        else:
            pgs = self.score(self.betas)

        # Sample the environmental/residual component:
        e = np.random.normal(loc=0., scale=np.sqrt(1. - self.h2g), size=self.N)

        # The final simulated phenotype is a combination of
        # the polygenic score + the residual component:
        y = pgs + e

        if self.phenotype_likelihood == 'binomial':
            # If the simulated phenotype is to be binary,
            # use the threshold model to determine positives/negatives
            # based on the prevalence of the phenotype in the population:
            cutoff = norm.ppf(1. - self.prevalence)
            new_y = np.zeros_like(y, dtype=int)
            new_y[y > cutoff] = 1
            self.phenotypes = new_y
        else:
            self.phenotypes = y

        return self.phenotypes

    def simulate(self, reset_beta=True, perform_gwas=False, phenotype_id=None):

        if self.betas is None or reset_beta:
            self.simulate_mixture_assignment()
            self.set_per_snp_heritability()
            self.simulate_betas()

        # Simulate the phenotype
        self.simulate_phenotypes()

        if perform_gwas:
            # Perform GWAS
            if self.use_plink:
                self.perform_gwas_plink()
            else:
                self.perform_gwas()

        if phenotype_id is not None:
            self.phenotype_id = phenotype_id

    def to_true_beta_table(self, per_chromosome=False):
        """
        Export the simulated true effect sizes and causal status
         into a pandas table.
        :param per_chromosome: If True, return a dictionary of tables for each chromosome.
        """

        assert self.betas is not None

        eff_tables = {}
        causal_status = self.get_causal_status()

        for c in self.chromosomes:

            eff_tables[c] = pd.DataFrame({
                'CHR': c,
                'SNP': self.snps[c],
                'A1': self.alt_alleles[c],
                'MixtureComponent': np.where(self.mixture_assignment[c] == 1)[1],
                'Heritability': self.per_snp_h2g[c],
                'BETA': self.betas[c].flatten(),
                'Causal': causal_status[c],
            })

        if per_chromosome:
            return eff_tables
        else:
            return pd.concat(list(eff_tables.values()))
