import os.path as osp
import pandas as pd
import numpy as np
import warnings
from magenpy.SumstatsTable import SumstatsTable


def perform_gwa_plink2(genotype_matrix,
                       temp_dir='temp',
                       standardize_phenotype=True,
                       include_covariates=True):

    from magenpy.GenotypeMatrix import plinkBEDGenotypeMatrix
    from magenpy.utils.executors import plink2Executor

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink2 = plink2Executor()

    s_table = genotype_matrix.sample_table

    if s_table.phenotype_likelihood is None:
        warnings.warn("The phenotype likelihood is not specified! "
                      "Assuming that the phenotype is continuous...")

    # Output phenotype table:
    phe_fname = osp.join(temp_dir, "pheno.txt")
    phe_table = s_table.get_phenotype_table()
    if s_table.phenotype_likelihood == 'binomial':
        phe_table['phenotype'] += 1
    phe_table.to_csv(phe_fname, sep="\t", index=False, header=False)

    # Process covariates:
    if include_covariates and s_table.covariates is not None:
        covar_fname = osp.join(temp_dir, "covar.txt")
        covar = s_table.get_covariates_table()
        covar.to_csv(covar_fname, sep="\t", index=False, header=False)
        covar_modifier = ''
    else:
        covar_fname = None
        covar_modifier = ' allow-no-covars'

    # Determine regression type based on phenotype likelihood:
    plink_reg_type = ['linear', 'logistic'][s_table.phenotype_likelihood == 'binomial']

    # Output subset of SNPs to perform association tests on:
    snp_keepfile = osp.join(temp_dir, f"variants.keep")
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    # Output file:
    plink_output = osp.join(temp_dir, f"output")

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--extract {snp_keepfile}",
        f"--{plink_reg_type} hide-covar{covar_modifier} cols=chrom,pos,alt1,ref,a1freq,nobs,beta,se,tz,p",
        f"--pheno {phe_fname}",
        f"--out {plink_output}"
    ]

    if standardize_phenotype and plink_reg_type == 'linear':
        cmd.append('--variance-standardize')

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
    ss_table = SumstatsTable.from_file(output_fname, sumstats_format='plink')
    # Make sure that the effect allele is encoded properly:
    ss_table.match(genotype_matrix.snp_table, correct_flips=True)

    return ss_table


def perform_gwa_plink1p9(genotype_matrix, standardize_phenotype=False):
    """
    TODO: Add support for performing association testing with plink1.9
    """

    #assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)
    raise NotImplementedError


def perform_gwa_xarray(genotype_matrix,
                       standardize_genotype=True,
                       standardize_phenotype=True):

    from magenpy.GenotypeMatrix import xarrayGenotypeMatrix
    assert isinstance(genotype_matrix, xarrayGenotypeMatrix)

    if genotype_matrix.sample_table.phenotype_likelihood is None:
        warnings.warn("The phenotype likelihood is not specified! "
                      "Assuming that the phenotype is continuous...")
    elif genotype_matrix.sample_table.phenotype_likelihood == 'binomial':
        raise Exception("The xarray backend does not support performing association "
                        "testing on binary (case-control) phenotypes!")

    sumstats_table = genotype_matrix.get_snp_table(
        ['CHR', 'SNP', 'POS', 'A1', 'A2', 'N', 'MAF']
    )

    phenotype = genotype_matrix.sample_table.phenotype

    if standardize_phenotype:
        from ..transforms.phenotype import standardize
        phenotype = standardize(phenotype)

    sigma_sq_y = np.var(phenotype)

    if standardize_genotype:

        from ..transforms.genotype import standardize

        sumstats_table['BETA'] = np.dot(standardize(genotype_matrix.xr_mat).T, phenotype) / sumstats_table['N'].values
        sumstats_table['SE'] = np.sqrt(sigma_sq_y / sumstats_table['N'].values)
    else:

        sumstats_table['BETA'] = (
            np.dot(genotype_matrix.xr_mat.fillna(sumstats_table['MAF'].values).T, phenotype) /
            sumstats_table['N'].values * genotype_matrix.maf_var
        )

        sumstats_table['SE'] = np.sqrt(sigma_sq_y / (sumstats_table['N'].values * genotype_matrix.maf_var))

    ss_table = SumstatsTable(sumstats_table)
    _, _ = ss_table.z_score, ss_table.pval

    return ss_table
