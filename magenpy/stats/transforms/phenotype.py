import numpy as np


def adjust_for_covariates(phenotype, covariates):
    """
    This function takes a phenotype vector and a matrix of covariates
    and applies covariate correction on the phenotype. Concretely,
    this involves fitting a linear model where the response is the
    phenotype and the predictors are the covariates and then returning
    the residuals.
    :param phenotype: A vector of continuous or quantitative phenotypes.
    :param covariates: A matrix where each row corresponds to an individual
     and each column corresponds to a covariate (e.g. age, sex, PCs, etc.)
    """

    import statsmodels.api as sm

    return sm.OLS(phenotype, sm.add_constant(covariates)).fit().resid


def rint(phenotype, offset=3./8):
    """
    Apply Rank-based inverse normal transform on the phenotype.
    :param phenotype: A vector of continuous or quantitative phenotypes.
    :param offset: The offset to use in the INT transformation (Blom's offset by default).
    """

    from scipy.stats import rankdata, norm

    ranked_pheno = rankdata(phenotype, method="average")
    return norm.ppf((ranked_pheno - offset) / (len(ranked_pheno) - 2 * offset + 1))


def find_outliers(phenotype, sigma_threshold=5):
    """
    Detect samples with outlier phenotype values.
    This function takes a vector of quantitative phenotypes,
    computes the z-score for every individual, and returns a
    boolean vector indicating whether individual i has phenotype value
    within the specified standard deviations `sigma_threshold`.
    :param sigma_threshold: The multiple of standard deviations or sigmas after
    which we consider the phenotypic value an outlier.
    """
    from scipy.stats import zscore
    return np.abs(zscore(phenotype)) < sigma_threshold


def standardize(phenotype):
    """
    Standardize the phenotype vector to have mean zero and unit variance
    :param phenotype: A numpy vector of continuous or quantitative phenotypes.
    """
    return (phenotype - phenotype.mean()) / phenotype.std()
