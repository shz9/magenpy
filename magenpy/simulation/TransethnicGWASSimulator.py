"""
Author: Shadi Zabad
Date: March 2021
"""

from magenpy.utils.system_utils import makedir
import os.path as osp
import pandas as pd
import numpy as np
from magenpy import GWASDataLoader
from magenpy.simulation.GWASSimulator import GWASSimulator


class TransethnicGWASSimulator(GWASDataLoader):
    """
    A module for simulating trans-ethnic GWAS data
    """

    def __init__(self, bed_files,
                 cluster_assignments,
                 cluster_header=None,
                 cluster_col=2,
                 p_shared_causal=1.,
                 rho=1.,
                 **kwargs):

        super().__init__(bed_files, **kwargs)

        self.cluster_table = self.read_cluster_assignments(cluster_assignments,
                                                           header=cluster_header,
                                                           cluster_col=cluster_col)

        self.ind_keep_files, self.keep_snps_file = self.create_cluster_keep_files()

        # Update the arguments:
        kwargs.update(keep_snps=self.keep_snps_file)

        # Proportion of causal snps that are shared
        self.p_shared_causal = p_shared_causal

        # Rho can be either a scalar or a matrix that determines the patterns of
        # correlations between effect sizes in different populations.
        if type(rho) == np.float:
            self.rho = rho*np.ones(shape=(len(self.clusters), len(self.clusters)))
            np.fill_diagonal(self.rho, 1.)
        else:
            self.rho = rho

        # A flag that declares whether betas have been simulated
        self.simulated_betas = False

        # Reference population
        self.ref_pop = None

        # A dictionary of GWAS simulators for each cluster
        self.cluster_simulators = {}

        for c in self.clusters:
            if self.ref_pop is None:
                self.ref_pop = c

            self.cluster_simulators[c] = GWASSimulator(bed_files,
                                                       keep_individuals=self.ind_keep_files[c],
                                                       **kwargs)

    @property
    def clusters(self):
        return self.cluster_table['Cluster'].unique()

    def set_reference_population(self, c):
        self.ref_pop = c

    def read_cluster_assignments(self, cluster_file, header=None, cluster_col=2):

        try:
            clusters = pd.read_csv(cluster_file, sep="\s+", header=header)
            clusters.columns = ['FID', 'IID'] + list(clusters.columns[2:])
            clusters = clusters[['FID', 'IID'] + [clusters.columns[cluster_col]]]
            clusters.columns = ['FID', 'IID', 'Cluster']
        except Exception as e:
            raise e

        return clusters

    def create_cluster_keep_files(self):

        keep_file_dir = osp.join(self.temp_dir, "cluster_keep_files")
        makedir(keep_file_dir)

        ind_keep_files = {}
        invar_snps = []

        for c in self.clusters:

            f_name = osp.join(keep_file_dir, str(c) + ".txt")

            c_members = self.cluster_table.loc[self.cluster_table['Cluster'] == c, 'IID']

            # Find all SNPs that are invariant within any of the clusters:
            for ch, gt in self.genotypes.items():
                # Note: Checking for invariant sites using var(col) == 0. can be unstable
                # numerically. Therefore, we use min(col) == max(col) instead.
                filt_subset = gt.sel(sample=c_members.values)
                invar_snps += list(self.snps[ch][np.where(filt_subset.min(axis=0) == filt_subset.max(axis=0))[0]])

            c_members.to_csv(
                f_name, index=False, header=False
            )

            ind_keep_files[c] = f_name

        # Save the common SNPs that harbor variation in all clusters:
        snps_to_keep = list(set(np.concatenate(list(self.snps.values()))) - set(invar_snps))

        self.filter_snps(snps_to_keep)

        keep_snps_file = osp.join(keep_file_dir, "keep_snps.txt")
        pd.Series(snps_to_keep).to_csv(keep_snps_file, index=False, header=False)

        return ind_keep_files, keep_snps_file

    def simulate_causal_status(self):

        # The reference simulator:
        ref_sim = self.cluster_simulators[self.ref_pop]

        # Simulate causal snps in reference population:
        ref_sim.simulate_mixture_assignment()

        # Get the causal snps in reference population:
        ref_causal = {
            c: np.where(a)[0]
            for c, a in ref_sim.get_causal_status().items()
        }

        for c in self.clusters:
            # For each population that is not the reference,
            # update their causal snps according to our draw for
            # the reference population
            if c != self.ref_pop:

                new_mixture = ref_sim.mixture_assignment.copy()

                if self.p_shared_causal < 1.:
                    for ch, ref_c in ref_causal.items():

                        # The number of shared causal snps for Chromosome `ch`:
                        n_shared_causal = int(np.floor(self.p_shared_causal * len(ref_c)))

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

                self.cluster_simulators[c].update_mixture_assignment(
                    new_mixture
                )

    def simulate_beta(self):

        for c in self.clusters:
            self.cluster_simulators[c].betas = {}

        for ch, c_size in self.shapes.items():
            # Draw the beta from a multivariate normal distribution with covariance
            # as specified in the matrix `rho`.
            betas = np.random.multivariate_normal(np.zeros(self.rho.shape[0]), cov=self.rho, size=c_size)
            for i, c in enumerate(self.clusters):
                self.cluster_simulators[c].betas[ch] = (
                        self.cluster_simulators[c].get_causal_status()[ch].astype(np.int)*betas[:, i]
                )

    def simulate(self, reset_beta=False, perform_gwas=True, phenotype_id=None):

        if reset_beta or not self.simulated_betas:
            self.simulate_causal_status()
            self.simulate_beta()
            self.simulated_betas = True

        self.phenotypes = pd.Series(np.zeros_like(self._iid), index=self._iid)

        for c in self.clusters:
            self.cluster_simulators[c].simulate(reset_beta=False)
            self.phenotypes[self.cluster_simulators[c].sample_ids] = self.cluster_simulators[c].phenotypes

        self.phenotypes = self.phenotypes.values.astype(np.float)

        # Perform GWAS on the pooled sample:
        if perform_gwas:
            self.perform_gwas()
