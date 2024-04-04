
import warnings
import numpy as np
import pandas as pd
from ..GWADataLoader import GWADataLoader


class PhenotypeSimulator(GWADataLoader):
    """
    A wrapper class that supports simulating complex traits with a variety of
    genetic architectures and heritability values, using the standard linear model. The
    basic implementation supports simulating effect sizes from a sparse Gaussian mixture density,
    allowing some variants to have larger effects than others. The class also supports simulating
    binary phenotypes (case-control) by thresholding the continuous phenotype at a specified threshold.

    To be concrete, the generative model for the simulation is as follows:

    1) Simulate the mixture assignment for each variant based on the mixing proportions `pi`.
    2) Simulate the effect sizes for each variant from the corresponding Gaussian density that they were assigned.
    3) Compute the polygenic score for each individual based on the simulated effect sizes.
    4) Simulate the residual component of the phenotype, in such a way that the total heritability is preserved.

    !!! seealso "See Also"
        * [GWADataLoader][magenpy.GWADataLoader.GWADataLoader]

    :ivar pi: The mixing proportions for the Gaussian mixture density.
    :ivar h2: The trait SNP heritability, or proportion of variance explained by SNPs.
    :ivar d: The variance multipliers for each component of the Gaussian mixture density.
    :ivar prevalence: The (disease) prevalence for binary (case-control) phenotypes.
    :ivar per_snp_h2: The per-SNP heritability for each variant in the dataset.
    :ivar per_snp_pi: The per-SNP mixing proportions for each variant in the dataset.
    :ivar beta: The effect sizes for each variant in the dataset.
    :ivar mixture_assignment: The assignment of each variant to a mixture component.

    """

    def __init__(self,
                 bed_files,
                 h2=0.2,
                 pi=0.1,
                 d=(0., 1.),
                 prevalence=0.15,
                 **kwargs):
        """
        Initialize the PhenotypeSimulator object with the necessary parameters.

        :param bed_files: A path (or list of paths) to PLINK BED files containing the genotype information.
        :param h2: The trait SNP heritability, or proportion of variance explained by SNPs.
        :param pi: The mixing proportions for the mixture of Gaussians (our model for the distribution of effect sizes).
        If a float is provided, it is converted to a tuple (1-pi, pi), where pi is the proportion of causal variants.
        :param d:  The variance multipliers for each component of the Gaussian mixture density. By default,
        all components have the same variance multiplier.
        :param prevalence: The (disease) prevalence for binary (case-control) phenotypes.
        """

        super().__init__(bed_files, **kwargs)

        # If pi is float, convert it to a tuple:
        if isinstance(pi, float):
            pi = (1. - pi, pi)

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
    def n_components(self):
        """
        :return: The number of Gaussian mixture components for the effect size distribution.
        """
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
        Set the per-SNP mixing proportions for each variant in the dataset.
        This is a convenience method that may come in handy for more flexible generative models.
        """

        self.per_snp_pi = {}

        for c, c_size in self.shapes.items():
            self.per_snp_pi[c] = np.repeat(np.array([self.pi]), c_size, axis=0)

    def set_per_snp_heritability(self):
        """
        Set the per-SNP heritability (effect size variance) for each variant in the dataset.
        This is a convenience method that may come in handy for more flexible generative models.
        """

        assert self.mixture_assignment is not None

        # Estimate the global sigma_beta_sq based on the
        # pre-specified heritability, the mixture proportions `pi`,
        # and the prior multipliers `d`.

        combined_assignments = np.concatenate([self.mixture_assignment[c] for c in self.chromosomes])

        sigma_beta_sq = self.h2 / (combined_assignments*self.d).sum()

        self.per_snp_h2 = {}

        for c, c_size in self.shapes.items():
            self.per_snp_h2[c] = sigma_beta_sq*self.d[np.where(self.mixture_assignment[c])[1]]

    def get_causal_status(self):
        """
        :return: A dictionary where the keys are the chromosome numbers
        and the values are binary vectors indicating which SNPs are
        causal for the simulated phenotype.

        :raises AssertionError: If the mixture assignment is not set.
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

    def set_causal_snps(self, causal_snps):
        """
        A utility method to set the causal variants in the simulation based on an array or
        list of SNPs specified by the user. The method takes an iterable (e.g. list or array) of `causal_snps`
        and then creates a new mixture assignment object where only the `causal_snps`
        contribute to the phenotype.

        :param causal_snps: A list or array of SNP rsIDs.
        :raises ValueError: If all mixture components are causal.

        """

        # Get the index of the mixture component whose multiplier is zero (i.e. the null component):
        try:
            zero_index = list(self.d).index(0)
        except ValueError:
            raise ValueError("Cannot set causal variants when all mixture components are causal. Modify "
                             "the mixture multipliers `d` to enable this functionality.")

        # Get the indices of the non-null mixture components:
        nonzero_indices = [i for i, d in enumerate(self.d) if d != 0.]

        # Get the mixture proportions for the non-null components and normalize them so they sum to 1:
        pis = np.array(self.pi)[nonzero_indices]
        pis /= pis.sum()

        # Initialize new mixture assignment object:
        new_assignment = {c: np.zeros((s, self.n_components)) for c, s in self.shapes.items()}

        from ..utils.compute_utils import intersect_arrays

        n_causal_set = 0

        for c, snps in self.snps.items():

            # Intersect the list of causal SNPs with the SNPs on chromosome `c`:
            snp_idx = intersect_arrays(snps, causal_snps, return_index=True)

            if len(snp_idx) > 0:
                n_causal_set += len(snp_idx)
                # For the causal SNPs, assign them randomly to the causal components
                new_assignment[c][snp_idx, np.random.choice(nonzero_indices,
                                                            size=len(snp_idx),
                                                            p=pis)] = 1
                # For the remaining SNPs, assign them to the null component:
                new_assignment[c][:, zero_index] = np.abs(new_assignment[c].sum(axis=1) - 1)

        if n_causal_set < len(causal_snps):
            warnings.warn("Not all causal SNPs are represented in the genotype matrix! "
                          f"User passed a list of {len(causal_snps)} SNPs, only matched {n_causal_set}.")

        self.set_mixture_assignment(new_assignment)

    def set_mixture_assignment(self, new_assignment):
        """
        Set the mixture assignments according to user-provided dictionary. The mixture
        assignment indicates which mixture component the effect size of a particular
        variant comes from.
        :param new_assignment: A dictionary where the keys are the chromosomes and
        the values are the mixture assignment for each SNP on that chromosome.
        """

        # Check that the shapes match pre-specified information:
        for c, c_size in self.shapes.items():
            assert new_assignment[c].shape == (c_size, self.n_components)

        self.mixture_assignment = new_assignment

    def simulate_mixture_assignment(self):
        """
        Simulate assigning SNPs to the various mixture components
        with probabilities given by mixing proportions `pi`.
        """

        if self.per_snp_pi is None or len(self.per_snp_pi) < 1:
            self.set_per_snp_mixture_probability()

        self.mixture_assignment = {}

        from ..utils.model_utils import multinomial_rvs

        for c, c_size in self.shapes.items():

            self.mixture_assignment[c] = multinomial_rvs(1, self.per_snp_pi[c])

        return self.mixture_assignment

    def set_beta(self, new_beta):
        """
        Set the variant effect sizes (beta) according to user-provided dictionary.

        :param new_beta: A dictionary where the keys are the chromosomes and
        the values are the beta (effect size) for each SNP on that chromosome.
        """

        # Check that the shapes match pre-specified information:
        for c, c_size in self.shapes.items():
            assert len(new_beta[c]) == c_size

        self.beta = new_beta

    def simulate_beta(self):
        """
        Simulate the causal effect size for variants included
        in the dataset. Here, the variant effect size is drawn from
        a Gaussian density with mean zero and scale given by
        the root of per-SNP heritability.
        """

        if self.per_snp_h2 is None or len(self.per_snp_h2) < 1:
            self.set_per_snp_heritability()

        self.beta = {}

        for c, c_size in self.shapes.items():

            self.beta[c] = np.random.normal(loc=0.0,
                                            scale=np.sqrt(self.per_snp_h2[c]),
                                            size=c_size)

        return self.beta

    def simulate_phenotype(self):
        """
        Simulate complex phenotypes for the samples present in the genotype matrix, given their
        genotype information and fixed effect sizes `beta` that were simulated previous steps.

        Given the simulated effect sizes, the phenotype is generated as follows:

        `Y = XB + e`

        Where `Y` is the vector of phenotypes, `X` is the genotype matrix, `B` is the vector of effect sizes,
        and `e` represents the residual effects.
        """

        assert self.beta is not None

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

            from scipy.stats import norm
            
            cutoff = norm.ppf(1. - self.prevalence)
            new_y = np.zeros_like(y, dtype=int)
            new_y[y > cutoff] = 1
        else:
            new_y = y

        self.set_phenotype(new_y)
        
        return new_y

    def simulate(self,
                 reset_beta=True,
                 reset_mixture_assignment=True,
                 perform_gwas=False):
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
        :param reset_mixture_assignment: If True, reset the assignment of SNPs to mixture components. Set to False
        if you'd like to keep the same configuration of causal SNPs.
        :param perform_gwas: If True, automatically perform genome-wide association on the newly simulated phenotype.
        """

        # Simulate the mixture assignment:
        if self.mixture_assignment is None or reset_mixture_assignment:
            self.simulate_mixture_assignment()

        # Set the per-SNP heritability based on the mixture assignment:
        self.set_per_snp_heritability()

        # Simulate betas based on per-SNP heritability
        if self.beta is None or reset_beta:
            self.simulate_beta()

        # Simulate the phenotype
        self.simulate_phenotype()

        if perform_gwas:
            # Perform genome-wide association testing...
            self.perform_gwas()

    def to_true_beta_table(self, per_chromosome=False):
        """
        Export the simulated true effect sizes and causal status into a pandas dataframe.
        :param per_chromosome: If True, return a dictionary of tables for each chromosome separately.

        :return: A pandas DataFrame with the true effect sizes and causal status for each variant.
        """

        assert self.beta is not None

        eff_tables = {}
        causal_status = self.get_causal_status()

        for c in self.chromosomes:

            eff_tables[c] = pd.DataFrame({
                'CHR': c,
                'SNP': self.genotype[c].snps,
                'A1': self.genotype[c].a1,
                'A2': self.genotype[c].a2,
                'MixtureComponent': np.where(self.mixture_assignment[c] == 1)[1],
                'Heritability': self.per_snp_h2[c],
                'BETA': self.beta[c].flatten(),
                'Causal': causal_status[c],
            })

        if per_chromosome:
            return eff_tables
        else:
            return pd.concat(list(eff_tables.values()))
