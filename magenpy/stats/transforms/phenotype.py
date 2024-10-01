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

     :return: The residuals of the linear model fit.
    """

    import statsmodels.api as sm

    return sm.OLS(phenotype, sm.add_constant(covariates)).fit().resid


def rint(phenotype, offset=3./8):
    """
    Apply Rank-based inverse normal transform on the phenotype.
    :param phenotype: A vector of continuous or quantitative phenotypes.
    :param offset: The offset to use in the INT transformation (Blom's offset by default).

    :return: The RINT-transformed phenotype.
    """

    from scipy.stats import rankdata, norm

    ranked_pheno = rankdata(phenotype, method="average")
    return norm.ppf((ranked_pheno - offset) / (len(ranked_pheno) - 2 * offset + 1))


def detect_outliers(phenotype, sigma_threshold=3, nan_policy='omit'):
    """
    Detect samples with outlier phenotype values.
    This function takes a vector of quantitative phenotypes,
    computes the z-score for every individual, and returns a
    boolean vector indicating whether individual i has phenotype value
    within the specified standard deviations `sigma_threshold`.
    :param phenotype: A numpy vector of continuous or quantitative phenotypes.
    :param sigma_threshold: The multiple of standard deviations or sigmas after
    which we consider the phenotypic value an outlier. Default is 3.
    :param nan_policy: The policy to use when encountering NaN values in the phenotype vector.
    By default, we compute the z-scores ignoring NaN values.

    :return: A boolean array indicating whether the phenotype value is an outlier (i.e.
    True indicates outlier).
    """
    from scipy.stats import zscore
    return np.abs(zscore(phenotype, nan_policy=nan_policy)) > sigma_threshold


def standardize(phenotype):
    """
    Standardize the phenotype vector to have mean zero and unit variance
    :param phenotype: A numpy vector of continuous or quantitative phenotypes.

    :return: The standardized phenotype array.
    """
    return (phenotype - phenotype.mean()) / phenotype.std()


def chained_transform(sample_table,
                      adjust_covariates=False,
                      standardize_phenotype=False,
                      rint_phenotype=False,
                      outlier_sigma_threshold=None,
                      transform_order=('standardize', 'covariate_adjust', 'rint', 'outlier_removal')):
    """
    Apply a chain of transformations to the phenotype vector.
    :param sample_table: An instance of SampleTable that contains phenotype information and other
    covariates about the samples in the dataset.
    :param adjust_covariates: If true, regress out the covariates from the phenotype. By default, we regress out all
    the covariates present in the SampleTable.
    :param standardize_phenotype: If true, standardize the phenotype.
    :param rint_phenotype: If true, apply Rank-based inverse normal transform.
    :param outlier_sigma_threshold: The multiple of standard deviations or sigmas after
    which we consider the phenotypic value an outlier.
    :param transform_order: A tuple specifying the order in which to apply the transformations. By default,
    the order is standardize, covariate_adjust, rint, and outlier_removal.

    :return: The transformed phenotype vector and a boolean mask indicating the samples that were not removed.
    """

    phenotype = sample_table.phenotype
    mask = np.ones_like(phenotype, dtype=bool)

    if sample_table.phenotype_likelihood != 'binomial':
        for transform in transform_order:

            if transform == 'standardize':
                # Standardize the phenotype:
                if standardize_phenotype:
                    phenotype = standardize(phenotype)

            elif transform == 'covariate_adjust':
                # Adjust the phenotype for a set of covariates:
                if adjust_covariates:
                    phenotype = adjust_for_covariates(phenotype, sample_table.get_covariates_matrix()[mask, :])

            elif transform == 'rint':
                # Apply Rank-based inverse normal transform (RINT) to the phenotype:
                if rint_phenotype:
                    phenotype = rint(phenotype)

            elif transform == 'outlier_removal':
                # Remove outlier samples whose phenotypes are more than `threshold` standard deviations from the mean:
                if outlier_sigma_threshold is not None:
                    # Find outliers:
                    mask = ~detect_outliers(phenotype, outlier_sigma_threshold)
                    # Filter phenotype vector to exclude outliers:
                    phenotype = phenotype[mask]

    return phenotype, mask
