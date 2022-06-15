import os.path as osp
import pandas as pd
from magenpy.utils.executors import plink2Executor
from magenpy.GenotypeMatrix import plinkBEDGenotypeMatrix
from magenpy.utils.model_utils import merge_snp_tables


def compute_allele_frequency_plink2(genotype_matrix, temp_dir='temp'):

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink2 = plink2Executor()

    s_table = genotype_matrix.sample_table

    keep_file = osp.join(temp_dir, 'samples.keep')
    keep_table = s_table.get_individual_table()
    keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

    snp_keepfile = osp.join(temp_dir, f"variants.keep")
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    plink_output = osp.join(temp_dir, "variants")

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--keep {keep_file}",
        f"--extract {snp_keepfile}",
        f"--freq",
        f"--out {plink_output}",
    ]

    plink2.execute(cmd)

    freq_df = pd.read_csv(plink_output + ".afreq", delim_whitespace=True)
    freq_df.rename(columns={'ID': 'SNP', 'ALT': 'A1', 'ALT_FREQS': 'MAF'}, inplace=True)
    merged_df = merge_snp_tables(genotype_matrix.get_snp_table(['SNP', 'A1']), freq_df)

    if len(merged_df) != genotype_matrix.n_snps:
        raise ValueError("Length of allele frequency table does not match number of SNPs.")

    return merged_df['MAF'].values


def compute_sample_size_per_snp_plink2(genotype_matrix, temp_dir='temp'):

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink2 = plink2Executor()

    s_table = genotype_matrix.sample_table

    keep_file = osp.join(temp_dir, 'samples.keep')
    keep_table = s_table.get_individual_table()
    keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

    snp_keepfile = osp.join(temp_dir, f"variants.keep")
    pd.DataFrame({'SNP': genotype_matrix.snps}).to_csv(
        snp_keepfile, index=False, header=False
    )

    plink_output = osp.join(temp_dir, "variants")

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--keep {keep_file}",
        f"--extract {snp_keepfile}",
        f"--missing variant-only",
        f"--out {plink_output}",
    ]

    plink2.execute(cmd)

    miss_df = pd.read_csv(plink_output + ".vmiss", delim_whitespace=True)
    miss_df = pd.DataFrame({'ID': genotype_matrix.snps}).merge(miss_df)

    if len(miss_df) != genotype_matrix.n_snps:
        raise ValueError("Length of missingness table does not match number of SNPs.")

    return (miss_df['OBS_CT'] - miss_df['MISSING_CT']).values
