
import numpy as np
from .PhenotypeSimulator import PhenotypeSimulator


class AnnotatedPhenotypeSimulator(PhenotypeSimulator):
    """
    Simulate complex traits by incorporating genomic functional
    annotations into the mixture densities that govern the effect size
    of each variant on the trait.

    !!! warning
        This code is experimental and needs much further validation.

    """

    def __init__(self, bed_files, **kwargs):
        """
        Create an instance of the AnnotatedPhenotypeSimulator class.

        :param bed_files: A list of BED files that contain the genotype data.
        :param kwargs: Additional keyword arguments for the PhenotypeSimulator class.
        """

        super().__init__(bed_files, **kwargs)

        # For now, we will restrict to 2 mixture components.
        assert self.n_components == 2

        self.w_h2 = None  # The annotation weights for the per-SNP heritability
        self.w_pi = None  # The annotation weights for the per-SNP causal probability

    def set_w_h2(self, w_h2):
        """
        Set the annotation weights for the per SNP heritability
        :param w_h2: A vector of weights for each annotation.
        """

        assert len(w_h2) == self.n_annotations

        self.w_h2 = w_h2
        self.set_per_snp_heritability()

    def simulate_w_h2(self, enrichment=None):
        """
        Simulate the annotation weights for the per-SNP heritability
        """
        raise NotImplementedError

    def set_w_pi(self, w_pi):
        """
        Set the annotation weights for the per SNP causal probability
        :param w_pi: A vector of weights for each annotation.
        """

        assert len(w_pi) == self.n_annotations

        self.w_pi = w_pi
        self.set_per_snp_mixture_probability()

    def simulate_w_pi(self, enrichment=None):
        """
        Simulate the annotation weights for the per-SNP causal probability

        :param enrichment: A dictionary of enrichment values where the
        key is the annotation and the value is the enrichment
        """

        enrichment = enrichment or {}
        enr = []
        for annot in self.annotation[self.chromosomes[0]].annotations:
            try:
                enr.append(enrichment[annot])
            except KeyError:
                enr.append(1.)

        self.w_pi = np.log(np.array(enr))

    def set_per_snp_heritability(self):
        """
        Set the per-SNP heritability values using the annotation weights.
        """

        if self.w_h2 is None:
            return super().set_per_snp_heritability()

        self.per_snp_h2 = {}

        for c in self.chromosomes:
            self.per_snp_h2[c] = np.clip(np.dot(self.annotation[c].values(), self.w_h2),
                                         a_min=0., a_max=np.inf)

    def set_per_snp_mixture_probability(self):
        """
        Set the per-SNP mixture probabilities using the annotation weights.
        """

        if self.w_pi is None:
            return super().set_per_snp_mixture_probability()

        self.per_snp_pi = {}

        for c in self.chromosomes:
            prob = 1./(1. + np.exp(-np.dot(self.annotation[c].values(add_intercept=True),
                                           np.concatenate([[np.log(self.pi[1])], self.w_pi]))))
            self.per_snp_pi[c] = np.array([1. - prob, prob]).T

    def get_heritability_enrichment(self):
        """
        Estimate the enrichment of heritability per annotation.
        """

        tabs = self.to_true_beta_table(per_chromosome=True)
        total_heritability = sum([tab['Heritability'].sum() for c, tab in tabs.items()])

        heritability_per_binary_annot = {
            bin_annot: 0. for bin_annot in self.annotation[self.chromosomes[0]].binary_annotations
        }

        n_variants_per_binary_annot = {
            bin_annot: 0 for bin_annot in heritability_per_binary_annot
        }

        for c, c_size in self.shapes.items():
            for bin_annot in self.annotation[c].binary_annotations:
                annot_idx = self.annotation[c].get_binary_annotation_index(bin_annot)
                heritability_per_binary_annot[bin_annot] += tabs[c].iloc[np.array(annot_idx), :]['Heritability'].sum()
                n_variants_per_binary_annot[bin_annot] += len(annot_idx)

        return {
            bin_annot: (ba_h2/total_heritability) / (n_variants_per_binary_annot[bin_annot] / self.m)
            for bin_annot, ba_h2 in heritability_per_binary_annot.items()
        }
