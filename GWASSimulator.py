import numpy as np
import dask.array as da
from xarray import DataArray

from .GWASDataLoader import GWASDataLoader


class GWASSimulator(GWASDataLoader):

    def __init__(self, bed_files, h2g=0.2, pis=(0.9, 0.1), gammas=(0., 1.), **kwargs):

        super().__init__(bed_files, **kwargs)

        self.h2g = h2g
        self.pis = pis
        self.gammas = np.array(gammas)

        self.annotation_weights = None

        self.betas = None
        self.mixture_assignment = None

    def simulate_genotypes(self, n):

        self.sample_ids = 'HG' + np.arange(1, n+1).astype(str).astype(np.object)
        self.train_idx = self.test_idx = self.ld_subset_idx = np.arange(n)

        for i, ld_fac_list in self.ld_cholesky_factors.items():

            gen = None
            var_coords = self.genotypes[i]['G'].variant.coords

            for j, ld_fac in enumerate(ld_fac_list):

                _, p = ld_fac.shape

                ng_mat = da.array(ld_fac.dot(da.random.normal(size=(n, p)).T).T)
                ng_mat -= da.mean(ng_mat, axis=0)
                ng_mat /= da.std(ng_mat, axis=0)

                if j > 0:
                    gen = da.concatenate([gen, ng_mat], axis=1)
                else:
                    gen = ng_mat

            g = DataArray(gen, dims=["sample", "variant"], coords=[self.sample_ids, var_coords['variant'].values])
            sample = {'iid': ("sample", self.sample_ids),
                      'fid': ("sample", self.sample_ids)}
            g = g.assign_coords(**sample)
            g = g.assign_coords(var_coords)
            g.name = "genotype"

            self.genotypes[i]['G'] = g

        self.N = n

    def simulate_mixture_assignment(self):

        self.mixture_assignment = {}

        for i, g_data in self.genotypes.items():
            _, p = g_data['G'].shape
            self.mixture_assignment[i] = np.random.multinomial(1, self.pis, size=p)

        return self.mixture_assignment

    def simulate_betas(self):

        self.betas = {}

        for i, g_data in self.genotypes.items():

            _, p = g_data['G'].shape

            if self.annotation_weights is not None:
                std_beta = np.sqrt(np.absolute(np.dot(self.annotations[i], self.annotation_weights)))
            else:
                std_beta = 1.

            betas = np.random.normal(loc=0.0,
                                     scale=self.gammas[np.where(self.mixture_assignment[i])[1]]*std_beta,
                                     size=p)

            self.betas[i] = betas

    def simulate_annotation_weights(self):
        if self.C is not None:
            self.annotation_weights = np.random.normal(scale=1./self.M, size=self.C)

    def simulate_phenotypes(self):

        g_comp = da.zeros(shape=self.N)

        for chrom_id in self.genotypes:
            g_comp += da.dot(self.genotypes[chrom_id]['G'], self.betas[chrom_id])

        g_var = np.var(g_comp, ddof=1)
        e_var = g_var * ((1.0 / self.h2g) - 1.0)

        e = da.random.normal(0, np.sqrt(e_var), self.N)

        y = g_comp + e
        y -= y.mean()
        y /= y.std()

        self.phenotypes = y.compute()
        self.phenotype_id = 'Simulated_' + str(np.random.randint(1, 1000))

        return self.phenotypes

    def simulate(self, n=None, reset_beta=False, phenotype_id=None):

        if n is not None:
            if self.ld_cholesky_factors is None:
                self.compute_cholesky_factors()
            self.simulate_genotypes(n)

        if self.betas is None or reset_beta:
            self.simulate_mixture_assignment()
            self.simulate_annotation_weights()
            self.simulate_betas()

        self.simulate_phenotypes()
        self.compute_summary_statistics()

        if phenotype_id is not None:
            self.phenotype_id = phenotype_id

