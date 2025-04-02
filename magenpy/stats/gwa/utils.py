from typing import Union
import os.path as osp
import pandas as pd
import numpy as np
import warnings
from ...GWADataLoader import GWADataLoader
from ...SumstatsTable import SumstatsTable
from ..transforms.phenotype import chained_transform


def inflation_factor(sumstats_input: Union[GWADataLoader, SumstatsTable, np.array]):
    """
    Compute the genomic control (GC) inflation factor (also known as lambda)
    from GWAS summary statistics.

    The inflation factor can be used to detect and correct inflation in the test statistics.

    :param sumstats_input: The input can be one of three classes of objects: A GWADataLoader object,
    a SumstatsTable object, or a numpy array of chi-squared statistics to compute the inflation factor.

    :return: The inflation factor (lambda) computed from the chi-squared statistics.
    """

    if isinstance(sumstats_input, GWADataLoader):
        chisq = np.concatenate([ss.get_chisq_statistic() for ss in sumstats_input.sumstats_table.values()])
    elif isinstance(sumstats_input, SumstatsTable):
        chisq = sumstats_input.get_chisq_statistic()
    else:
        chisq = sumstats_input

    from scipy.stats import chi2

    return np.median(chisq) / chi2.median(1)


def perform_gwa_plink2(genotype_matrix,
                       temp_dir='temp',
                       **phenotype_transform_kwargs):
    """

    Perform genome-wide association testing using plink 2.0
    This function takes a GenotypeMatrix object and gwas-related flags and
    calls plink to perform GWA on the genotype and phenotype data referenced
    by the GenotypeMatrix object.

    :param genotype_matrix: A plinkBEDGenotypeMatrix object.
    :param temp_dir: Path to a directory where we keep intermediate temporary files from plink.
    :param phenotype_transform_kwargs: Keyword arguments to pass to the `chained_transform` function. These arguments
    include the following options to transform the phenotype before performing GWAS:
    `adjust_covariates`, `standardize_phenotype`, `rint_phenotype`, and `outlier_sigma_threshold`. NOTE: These
    transformations are only applied to continuous phenotypes (`likelihood='gaussian'`).

    :return: A SumstatsTable object containing the summary statistics from the association tests.
    """

    from ...GenotypeMatrix import plinkBEDGenotypeMatrix
    from ...utils.executors import plink2Executor

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink2 = plink2Executor()

    s_table = genotype_matrix.sample_table

    if s_table.phenotype_likelihood is None:
        warnings.warn("The phenotype likelihood is not specified! "
                      "Assuming that the phenotype is continuous...")

    # It can happen sometimes that some interfaces call this function
    # with the `standardize_genotype` flag set to True. We remove this
    # flag from the phenotype transformation kwargs to avoid errors:
    if 'standardize_genotype' in phenotype_transform_kwargs:
        del phenotype_transform_kwargs['standardize_genotype']

    # Transform the phenotype:
    phenotype, mask = chained_transform(s_table, **phenotype_transform_kwargs)

    # Prepare the phenotype table to pass to plink:
    phe_table = s_table.get_phenotype_table()

    # If the likelihood is binomial, transform the phenotype into
    # plink's coding for case/control (1/2) rather than (0/1).
    if s_table.phenotype_likelihood == 'binomial':
        phe_table['phenotype'] += 1
    else:
        phe_table = phe_table.loc[mask, :]
        phe_table['phenotype'] = phenotype

    # Output phenotype table:
    phe_fname = osp.join(temp_dir, "pheno.txt")
    phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

    # Process covariates:
    if s_table.phenotype_likelihood == 'binomial' and 'adjust_covariates' in phenotype_transform_kwargs and \
            phenotype_transform_kwargs['adjust_covariates']:

        covar_fname = osp.join(temp_dir, "covar.txt")
        covar = s_table.get_covariates_table().loc[mask, :]
        covar.to_csv(covar_fname, sep="\t", index=False, header=False)
        covar_modifier = ''
    else:
        covar_fname = None
        covar_modifier = ' allow-no-covars'

    # Determine regression type based on phenotype likelihood:
    plink_reg_type = ['linear', 'logistic'][s_table.phenotype_likelihood == 'binomial']

    # Output subset of SNPs to perform association tests on:
    snp_keepfile = osp.join(temp_dir, "variants.keep")
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    # Output file:
    plink_output = osp.join(temp_dir, "output")

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--extract {snp_keepfile}",
        f"--{plink_reg_type} hide-covar{covar_modifier} cols=chrom,pos,alt1,ref,a1freq,nobs,beta,se,tz,p",
        f"--pheno {phe_fname}",
        f"--out {plink_output}"
    ]

    if covar_fname is not None:
        cmd.append(f'--covar {covar_fname}')

    plink2.execute(cmd)

    output_fname = plink_output + f".PHENO1.glm.{plink_reg_type}"

    if not osp.isfile(output_fname):
        if plink_reg_type == 'logistic' and osp.isfile(output_fname + ".hybrid"):
            output_fname += ".hybrid"
        else:
            raise FileNotFoundError

    # Read the summary statistics file from plink:
    ss_table = SumstatsTable.from_file(output_fname, sumstats_format='plink2')
    # Make sure that the effect allele is encoded properly:
    ss_table.match(genotype_matrix.snp_table, correct_flips=True)

    return ss_table


def perform_gwa_plink1p9(genotype_matrix,
                         temp_dir='temp',
                         **phenotype_transform_kwargs):
    """
    Perform genome-wide association testing using plink 1.9
    This function takes a GenotypeMatrix object and gwas-related flags and
    calls plink to perform GWA on the genotype and phenotype data referenced
    by the GenotypeMatrix object.

    :param genotype_matrix: A plinkBEDGenotypeMatrix object.
    :param temp_dir: Path to a directory where we keep intermediate temporary files from plink.
    :param phenotype_transform_kwargs: Keyword arguments to pass to the `chained_transform` function. These arguments
    include the following options to transform the phenotype before performing GWAS:
    `adjust_covariates`, `standardize_phenotype`, `rint_phenotype`, and `outlier_sigma_threshold`. NOTE: These
    transformations are only applied to continuous phenotypes (`likelihood='gaussian'`).

    :return: A SumstatsTable object containing the summary statistics from the association tests.
    """

    from ...GenotypeMatrix import plinkBEDGenotypeMatrix
    from ...utils.executors import plink1Executor

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink1 = plink1Executor()

    s_table = genotype_matrix.sample_table

    if s_table.phenotype_likelihood is None:
        warnings.warn("The phenotype likelihood is not specified! "
                      "Assuming that the phenotype is continuous...")

    # Transform the phenotype:
    phenotype, mask = chained_transform(s_table, **phenotype_transform_kwargs)

    # Prepare the phenotype table to pass to plink:
    phe_table = s_table.get_phenotype_table()

    # If the likelihood is binomial, transform the phenotype into
    # plink's coding for case/control (1/2) rather than (0/1).
    if s_table.phenotype_likelihood == 'binomial':
        phe_table['phenotype'] += 1
    else:
        phe_table = phe_table.loc[mask, :]
        phe_table['phenotype'] = phenotype

    # Output phenotype table:
    phe_fname = osp.join(temp_dir, "pheno.txt")
    phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

    # Process covariates:
    if s_table.phenotype_likelihood == 'binomial' and 'adjust_covariates' in phenotype_transform_kwargs and \
            phenotype_transform_kwargs['adjust_covariates']:

        covar_fname = osp.join(temp_dir, "covar.txt")
        covar = s_table.get_covariates_table().loc[mask, :]
        covar.to_csv(covar_fname, sep="\t", index=False, header=False)
    else:
        covar_fname = None

    # Determine regression type based on phenotype likelihood:
    plink_reg_type = ['linear', 'logistic'][s_table.phenotype_likelihood == 'binomial']

    # Output subset of SNPs to perform association tests on:
    snp_keepfile = osp.join(temp_dir, "variants.keep")
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    # Output file:
    plink_output = osp.join(temp_dir, "output")

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--extract {snp_keepfile}",
        f"--{plink_reg_type} hide-covar",
        f"--pheno {phe_fname}",
        f"--out {plink_output}"
    ]

    if covar_fname is not None:
        cmd.append(f'--covar {covar_fname}')

    plink1.execute(cmd)

    output_fname = plink_output + f".PHENO1.assoc.{plink_reg_type}"

    if not osp.isfile(output_fname):
        if plink_reg_type == 'logistic' and osp.isfile(output_fname + ".hybrid"):
            output_fname += ".hybrid"
        else:
            raise FileNotFoundError

    # Read the summary statistics file from plink:
    ss_table = SumstatsTable.from_file(output_fname, sumstats_format='plink1.9')
    # Infer the reference allele:
    ss_table.infer_a2(genotype_matrix.snp_table)

    # Make sure that the effect allele is encoded properly:
    ss_table.match(genotype_matrix.snp_table, correct_flips=True)

    return ss_table


def perform_gwa_xarray(genotype_matrix,
                       standardize_genotype=False,
                       **phenotype_transform_kwargs):
    """
    Perform genome-wide association testing using xarray and the PyData ecosystem.
    This function takes a GenotypeMatrix object and gwas-related flags and
    calls performs (simple) GWA on the genotype and phenotype data referenced
    by the GenotypeMatrix object. This function only implements GWA testing for
    continuous phenotypes. For other functionality (e.g. case-control GWAS),
    please use `plink` as a backend or consult other GWAS software (e.g. GCTA or REGENIE).

    :param genotype_matrix: A GenotypeMatrix object.
    :param standardize_genotype: If True, the genotype matrix will be standardized such that the columns (i.e. SNPs)
    have zero mean and unit variance.
    :param phenotype_transform_kwargs: Keyword arguments to pass to the `chained_transform` function. These arguments
    include the following options to transform the phenotype before performing GWAS:
    `adjust_covariates`, `standardize_phenotype`, `rint_phenotype`, and `outlier_sigma_threshold`. NOTE: These
    transformations are only applied to continuous phenotypes (`likelihood='gaussian'`).

    :return: A SumstatsTable object containing the summary statistics from the association tests.
    """

    # Sanity checks:

    # Check that the genotype matrix is an xarrayGenotypeMatrix object.
    from ...GenotypeMatrix import xarrayGenotypeMatrix
    assert isinstance(genotype_matrix, xarrayGenotypeMatrix)

    # Check that the phenotype likelihood is set correctly and that the phenotype is continuous.
    if genotype_matrix.sample_table.phenotype_likelihood is None:
        warnings.warn("The phenotype likelihood is not specified! "
                      "Assuming that the phenotype is continuous...")
    elif genotype_matrix.sample_table.phenotype_likelihood == 'binomial':
        raise ValueError("The xarray backend currently does not support performing association "
                         "testing on binary (case-control) phenotypes! Try setting the backend to `plink` or "
                         "use external software (e.g. GCTA or REGENIE) for performing GWAS.")

    # -----------------------------------------------------------

    # Get the SNP table from the genotype_matrix object:
    sumstats_table = genotype_matrix.get_snp_table(
        ['CHR', 'SNP', 'POS', 'A1', 'A2']
    )

    # -----------------------------------------------------------

    # Transform the phenotype:
    phenotype, mask = chained_transform(genotype_matrix.sample_table, **phenotype_transform_kwargs)

    # -----------------------------------------------------------
    # Prepare the genotype data for association testing:

    # Apply the mask to the genotype matrix:
    xr_mat = genotype_matrix.xr_mat[mask, :]

    # Compute sample size per SNP:
    n_per_snp = xr_mat.shape[0] - xr_mat.isnull().sum(axis=0).compute().values

    # Compute minor allele frequency per SNP:
    maf = xr_mat.sum(axis=0).compute().values / (2 * n_per_snp)

    # Standardize or center the genotype matrix (account for missing values):
    if standardize_genotype:
        from ..transforms.genotype import standardize
        xr_mat = standardize(xr_mat)
    else:
        xr_mat = (xr_mat - 2.*maf)

    xr_mat = xr_mat.fillna(0.)

    # Compute the sum of squares per SNP:
    sum_x_sq = (xr_mat**2).sum(axis=0).compute().values

    # -----------------------------------------------------------
    # Compute quantities for association testing:

    slope = np.dot(xr_mat.T, phenotype - phenotype.mean()) / sum_x_sq
    intercept = phenotype.mean()

    y_hat = xr_mat*slope + intercept

    s2 = ((phenotype.reshape(-1, 1) - y_hat)**2).sum(axis=0) / (n_per_snp - 2)

    se = np.sqrt(s2 / sum_x_sq)

    # -----------------------------------------------------------
    # Populate the data in the summary statistics table:

    sumstats_table['MAF'] = maf
    sumstats_table['N'] = n_per_snp
    sumstats_table['BETA'] = slope
    sumstats_table['SE'] = se

    ss_table = SumstatsTable(sumstats_table)
    # Trigger computing z-score and p-values from the BETA and SE columns:
    _, _ = ss_table.z_score, ss_table.pval

    return ss_table
