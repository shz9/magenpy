
import pandas as pd
import numpy as np
from ..GWADataLoader import GWADataLoader
from .PhenotypeSimulator import PhenotypeSimulator


class MultiCohortPhenotypeSimulator(GWADataLoader):
    """
    A module for simulating GWAS data for separate cohorts or clusters of the data.
    This includes scenarios such as multi-population or multi-ethnic datasets, or 
    datasets that can be stratified by a discrete variable.

    !!! warning
        This code is experimental and needs much further validation.

    """

    def __init__(self,
                 bed_files,
                 cluster_assignments_file,
                 prop_shared_causal=1.,
                 rho=1.,
                 **kwargs):
        """
        Simulate phenotypes using the linear additive model while accounting 
        for heterogeneous genetic architectures across cohorts.
    
        :param bed_files: A path (or list of paths) to PLINK BED files.
        :param cluster_assignments_file: A file mapping each sample in the BED files to their corresponding 
        cohort or cluster.
        :param prop_shared_causal: Proportion of causal variants that are shared across clusters.
        :param rho: The correlation coefficient for the effect size across clusters.
        """

        super().__init__(bed_files, **kwargs)

        from ..parsers.misc_parsers import parse_cluster_assignment_file

        self.cluster_table = parse_cluster_assignment_file(cluster_assignments_file)

        # Proportion of causal snps that are shared
        self.prop_shared_causal = prop_shared_causal

        # Rho can be either a scalar or a matrix that determines the patterns of
        # correlations between effect sizes in different clusters.
        if np.issubdtype(type(rho), np.floating):
            self.rho = rho*np.ones(shape=(len(self.clusters), len(self.clusters)))
            np.fill_diagonal(self.rho, 1.)
        else:
            self.rho = rho

        # Reference cluster
        self.ref_cluster = None

        # A dictionary of GWAS simulators for each cluster
        self.cluster_simulators = {}

        for c in self.clusters:
            if self.ref_cluster is None:
                self.ref_cluster = c

            self.cluster_simulators[c] = PhenotypeSimulator(bed_files,
                                                            keep_samples=self.get_samples_in_cluster(c),
                                                            **kwargs)

    @property
    def clusters(self):
        return self.cluster_table['Cluster'].unique()

    def get_samples_in_cluster(self, cluster):
        return self.cluster_table.loc[self.cluster_table['Cluster'] == cluster, 'IID'].values

    def set_reference_cluster(self, c):
        self.ref_cluster = c

    def simulate_causal_status(self):

        # The reference simulator:
        ref_sim = self.cluster_simulators[self.ref_cluster]

        # Simulate causal snps in reference cluster:
        ref_sim.simulate_mixture_assignment()

        # Get the causal snps in reference cluster:
        ref_causal = {
            c: np.where(a)[0]
            for c, a in ref_sim.get_causal_status().items()
        }

        for c in self.clusters:
            # For each cluster that is not the reference,
            # update their causal snps according to our draw for
            # the reference cluster
            if c != self.ref_cluster:

                new_mixture = ref_sim.mixture_assignment.copy()

                if self.prop_shared_causal < 1.:
                    for ch, ref_c in ref_causal.items():

                        # The number of shared causal snps for Chromosome `ch`:
                        n_shared_causal = int(np.floor(self.prop_shared_causal * len(ref_c)))

                        # Number of snps to flip:
                        n_flip = len(ref_c) - n_shared_causal

                        # Randomly decide which snps to "turn off":
                        for i in np.random.choice(ref_c, size=n_flip, replace=False):
                            new_mixture[ch][i] = new_mixture[ch][i][::-1]
                            # With probability p, switch on some other randomly chosen SNP:
                            # NOTE: If the number of SNPs is small, there's a small chance
                            # that this may flip the same SNP multiple times.
                            if np.random.uniform() < ref_sim.pis[1]:
                                new_i = np.random.choice(self.shapes[ch])
                                new_mixture[ch][new_i] = new_mixture[ch][new_i][::-1]

                self.cluster_simulators[c].set_mixture_assignment(
                    new_mixture
                )

    def simulate_beta(self):

        for c in self.clusters:
            self.cluster_simulators[c].beta = {}

        for ch, c_size in self.shapes.items():
            # Draw the beta from a multivariate normal distribution with covariance
            # as specified in the matrix `rho`.
            betas = np.random.multivariate_normal(np.zeros(self.rho.shape[0]), cov=self.rho, size=c_size)
            for i, c in enumerate(self.clusters):
                self.cluster_simulators[c].beta[ch] = (
                        self.cluster_simulators[c].get_causal_status()[ch].astype(np.int32)*betas[:, i]
                )

    def simulate(self, perform_gwas=False):

        self.simulate_causal_status()
        self.simulate_beta()

        iids = self.sample_table.iid

        phenotypes = pd.Series(np.zeros_like(iids), index=iids)

        for c in self.clusters:
            self.cluster_simulators[c].simulate(reset_beta=False)
            phenotypes[self.cluster_simulators[c].sample_table.iid] = self.cluster_simulators[c].sample_table.phenotype

        self.set_phenotype(phenotypes)

        # Perform GWAS on the pooled sample:
        if perform_gwas:
            self.perform_gwas()
