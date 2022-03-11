
import numpy as np
from .GWASSimulator import GWASSimulator


class AnnotatedGWASSimulator(GWASSimulator):

    def __init__(self, bed_files, **kwargs):
        super().__init__(bed_files, **kwargs)

        # For now, we will restrict to 2 mixture components.
        assert self.n_mixtures == 2

        self.w_theta = None
        self.w_phi = None

    def set_w_theta(self, w_theta):

        assert len(w_theta) == self.n_annotations

        self.w_theta = w_theta
        self.set_per_snp_heritability()

    def simulate_w_theta(self, enrichment=None):
        pass

    def set_w_phi(self, w_phi):

        assert len(w_phi) == self.n_annotations

        self.w_phi = w_phi
        self.set_per_snp_mixture_probability()

    def simulate_w_phi(self, enrichment=None):
        pass

    def set_per_snp_heritability(self):

        if self.w_theta is None:
            return super().set_per_snp_heritability()

        self.per_snp_h2g = {}

        for c in self.chromosomes:
            self.per_snp_h2g[c] = np.clip(np.dot(self.annotations[c].values(), self.w_theta), a_min=0., a_max=np.inf)

    def set_per_snp_mixture_probability(self):

        if self.w_phi is None:
            return super().set_per_snp_mixture_probability()

        self.per_snp_pi = {}

        for c in self.chromosomes:
            prob = 1./(1. + np.exp(-np.dot(self.annotations[c].values(), self.w_phi)))
            self.per_snp_pi[c] = np.array([1. - prob, prob]).T
            print(self.per_snp_pi[c])

    def get_heritability_enrichment(self):

        tabs = self.to_true_beta_table(per_chromosome=True)
        total_heritability = sum([tab['Heritability'].sum() for c, tab in tabs.items()])

        heritability_per_binary_annot = {
            bin_annot: 0. for bin_annot in self.annotations[self.chromosomes[0]].binary_annotations
        }

        n_variants_per_binary_annot = {
            bin_annot: 0 for bin_annot in heritability_per_binary_annot
        }

        for c, c_size in self.shapes.items():
            for bin_annot in self.annotations[c].binary_annotations:
                annot_idx = self.annotations[c].get_binary_annotation_index(bin_annot)
                heritability_per_binary_annot[bin_annot] += tabs[c].iloc[np.array(annot_idx), :]['Heritability'].sum()
                n_variants_per_binary_annot[bin_annot] += len(annot_idx)

        return {
            bin_annot: (ba_h2g/total_heritability) / (n_variants_per_binary_annot[bin_annot] / self.M)
            for bin_annot, ba_h2g in heritability_per_binary_annot.items()
        }
