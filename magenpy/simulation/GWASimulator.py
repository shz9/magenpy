"""
Author: Shadi Zabad
Date: March 2021
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from magenpy.utils.model_utils import multinomial_rvs
from magenpy.GWADataLoader import GWADataLoader


class GWASimulator(GWADataLoader):

    def __init__(self,
                 bed_files,
                 h2=0.2,
                 pi=(0.9, 0.1),
                 d=(0., 1.),
                 prevalence=0.15,
                 **kwargs):
        """
        Simulate phenotypes using the linear additive model.

        :param bed_files: A path (or list of paths) to PLINK BED files.
        :param h2: The trait SNP heritability, or proportion of variance explained by SNPs.
        :param pi: The mixture proportions for Gaussian mixture density.
        :param d:  The variance multipliers for each component of the Gaussian mixture density.
        :param prevalence: The (disease) prevalence for binary (case-control) phenotypes.
        """

        super().__init__(bed_files, **kwargs)

        self.pi = pi
        self.h2 = h2
        self.prevalence = prevalence

        # Sanity checks:
        assert 0. <= self.h2 <= 1.
        assert round(sum(self.pi), 1) == 1.
        assert 0. < self.prevalence < 1.

        self.d = np.array(d)

        self.per_snp_h2 = None
        self.per_snp_pi = None
        self.beta = None
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

    def set_h2(self, new_h2):
        """
        Set the total heritability (proportion of additive variance due to SNPs) for the trait
        """
        self.h2 = new_h2
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
        sigma_beta_sq = self.h2 / (self.m*np.array(self.pi)*self.d).sum()

        self.per_snp_h2 = {}

        for c, c_size in self.shapes.items():
            self.per_snp_h2[c] = sigma_beta_sq*self.d[np.where(self.mixture_assignment[c])[1]]

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

    def set_beta(self, new_beta):
        """
        Set the beta according to user-provided dictionary.
        :param new_beta: A dictionary where the keys are the chromosomes and
        the values are the beta for each SNP on that chromosome.
        """

        # Check that the shapes match pre-specified information:
        for c, c_size in self.shapes.items():
            assert len(new_beta[c]) == c_size

        self.beta = new_beta

    def simulate_beta(self):
        """
        Simulate the causal effect size for the variants included
        in the dataset.
        """

        if self.per_snp_h2 is None or len(self.per_snp_h2) < 1:
            self.set_per_snp_heritability()

        self.beta = {}

        for c, c_size in self.shapes.items():

            self.beta[c] = np.random.normal(loc=0.0,
                                            scale=np.sqrt(self.per_snp_h2[c]),
                                            size=c_size)

        return self.beta

    def simulate_phenotypes(self):
        """
        Simulate complex phenotypes for the samples, given their genotype information and 
        fixed effect sizes `beta` that were simulated previously.
        """

        assert self.beta

        # Compute the polygenic score given the simulated/provided beta:
        pgs = self.score(self.beta)

        # Sample the environmental/residual component:
        e = np.random.normal(loc=0., scale=np.sqrt(1. - self.h2), size=self.sample_size)

        # The final simulated phenotype is a combination of
        # the polygenic score + the residual component:
        y = pgs + e

        if self.phenotype_likelihood == 'binomial':
            # If the simulated phenotype is to be binary,
            # use the threshold model to determine positives/negatives
            # based on the prevalence of the phenotype in the population:
            
            from ..stats.transforms.phenotype import standardize
            
            y = standardize(y)
            
            cutoff = norm.ppf(1. - self.prevalence)
            new_y = np.zeros_like(y, dtype=int)
            new_y[y > cutoff] = 1
        else:
            new_y = y

        self.set_phenotype(new_y)
        
        return new_y

    def simulate(self, reset_beta=True, perform_gwas=False):
        """
        A convenience method to simulate all the components of the generative model.
        Specifically, the simulation follows the standard linear model, where the phenotype is 
        dependent on the genotype + environmental components that are assumed to be uncorrelated:
        
        `Y = XB + e`
        
        Where `Y` is the vector of phenotypes, `X` is the genotype matrix, `B` is the vector of effect sizes, 
        and `e` represents the residual effects. The generative model proceeds by:
         
         1) Drawing the effect sizes `beta` from a Gaussian mixture density.
         2) Drawing the residual effect from an isotropic Gaussian density.
         3) Setting the phenotype according to the equation above. 
        
        :param reset_beta: If True, reset the effect sizes by drawing new ones from the prior density.
        :param perform_gwas: If True, automatically perform genome-wide association on the newly simulated phenotype.
        """

        if self.beta is None or reset_beta:
            self.simulate_mixture_assignment()
            self.set_per_snp_heritability()
            self.simulate_beta()

        # Simulate the phenotype
        self.simulate_phenotypes()

        if perform_gwas:
            # Perform genome-wide association testing...
            self.perform_gwas()

    def to_true_beta_table(self, per_chromosome=False):
        """
        Export the simulated true effect sizes and causal status
         into a pandas table.
        :param per_chromosome: If True, return a dictionary of tables for each chromosome.
        """

        assert self.beta is not None

        eff_tables = {}
        causal_status = self.get_causal_status()

        for c in self.chromosomes:

            eff_tables[c] = pd.DataFrame({
                'CHR': c,
                'SNP': self.genotype[c].snps,
                'A1': self.genotype[c].a1,
                'MixtureComponent': np.where(self.mixture_assignment[c] == 1)[1],
                'Heritability': self.per_snp_h2[c],
                'BETA': self.beta[c].flatten(),
                'Causal': causal_status[c],
            })

        if per_chromosome:
            return eff_tables
        else:
            return pd.concat(list(eff_tables.values()))
