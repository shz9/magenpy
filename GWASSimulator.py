"""
Author: Shadi Zabad
Date: March 2021
"""

import numpy as np
from .GWASDataLoader import GWASDataLoader


class GWASSimulator(GWASDataLoader):

    def __init__(self, bed_files,
                 h2g=0.2,
                 pis=(0.9, 0.1),
                 gammas=(0., 1.),
                 binomial_threshold=0.,
                 **kwargs):

        super().__init__(bed_files, **kwargs)

        self.h2g = h2g
        self.pis = pis

        assert 0. <= self.h2g <= 1.
        assert sum(self.pis) == 1.

        self.gammas = np.array(gammas)
        self.binomial_threshold = binomial_threshold

        self.annotation_weights = None

        self.betas = None
        self.mixture_assignment = None

    def get_causal_status(self):
        """
        This method returns a dictionary of binary vectors
        indicating which snps are causal for each chromosome
        :return:
        """

        assert self.mixture_assignment is not None

        try:
            zero_index = list(self.gammas).index(0)
        except ValueError:
            # If all SNPs are causal:
            return {c: np.repeat(True, c_size) for c, c_size in self.shapes.items()}

        causal_status = {}

        for c, mix_a in self.mixture_assignment.items():
            causal_status[c] = np.where(mix_a)[1] != zero_index

        return causal_status

    def update_mixture_assignment(self, new_assignment):
        self.mixture_assignment = new_assignment

    def update_betas(self, new_betas):
        self.betas = new_betas

    def simulate_mixture_assignment(self):
        """
        Simulate assigning SNPs to the various mixture components
        with probabilities self.pis
        :return:
        """

        self.mixture_assignment = {}

        for c, c_size in self.shapes.items():
            if all([(i < 1.) for i in self.pis]):
                self.mixture_assignment[c] = np.random.multinomial(1, self.pis, size=c_size)
            else:
                # if all snps are assigned to one mixture (e.g. all snps are causal)
                assign = np.zeros(len(self.pis), dtype=np.int)
                assign[self.pis.index(1)] = 1
                self.mixture_assignment[c] = np.repeat(np.array([assign]), c_size, axis=0)

        return self.mixture_assignment

    def simulate_betas(self):
        """
        Simulate the causal effect size for the snps
        :return:
        """

        self.betas = {}

        for i, g_data in self.genotypes.items():

            _, p = g_data.shape

            if self.annotation_weights is not None:
                std_beta = np.sqrt(np.absolute(np.dot(self.annotations[i], self.annotation_weights)))
            else:
                std_beta = 1.

            betas = np.random.normal(loc=0.0,
                                     scale=self.gammas[np.where(self.mixture_assignment[i])[1]]*std_beta,
                                     size=p)

            self.betas[i] = betas

    def simulate_annotation_weights(self):
        """
        Simulate annotation weights, which would influence the variance in causal effect size
        :return:
        """
        if self.C is not None:
            self.annotation_weights = np.random.normal(scale=1./self.M, size=self.C)

    def simulate_phenotypes(self):
        """
        Simulate the phenotype for N individuals
        :return:
        """

        if self.use_plink:
            pgs = self.predict_plink(self.betas)
        else:
            pgs = self.predict(self.betas)

        # Estimate the genetic variance
        g_var = np.var(pgs, ddof=1)

        # If genetic variance > 0., assign environmental variance such that
        # ratio of genetic variance to total phenotypic variance is h2g
        if g_var > 0.:
            e_var = g_var * ((1.0 / self.h2g) - 1.0)
        else:
            e_var = 1.

        # Compute the environmental component:
        e = np.random.normal(0, np.sqrt(e_var), self.N)

        # Compute the simulated phenotype:
        y = pgs + e

        if self.standardize_phenotype or self.phenotype_likelihood == 'binomial':
            # Standardize if the trait is binary:
            y -= y.mean()
            y /= y.std()

        if self.phenotype_likelihood == 'binomial':
            y[y > self.binomial_threshold] = 1.
            y[y <= self.binomial_threshold] = 0.

        self.phenotypes = y

        return self.phenotypes

    def simulate(self, reset_beta=False, perform_gwas=False, phenotype_id=None):

        if self.betas is None or reset_beta:
            self.simulate_mixture_assignment()
            self.simulate_annotation_weights()
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

