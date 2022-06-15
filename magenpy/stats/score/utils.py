import os.path as osp
import numpy as np
import pandas as pd


def score_plink2(genotype_matrix,
                 betas,
                 standardize_genotype=False,
                 temp_dir='temp'):

    from magenpy.GenotypeMatrix import plinkBEDGenotypeMatrix
    from magenpy.utils.executors import plink2Executor

    assert isinstance(genotype_matrix, plinkBEDGenotypeMatrix)

    plink2 = plink2Executor()

    try:
        betas_shape = betas.shape[1]
        if betas_shape == 1:
            raise IndexError
        score_col_nums = f"--score-col-nums 3-{3 + betas_shape - 1}"
    except IndexError:
        betas_shape = 1
        betas = betas.reshape(-1, 1)
        score_col_nums = f"--score-col-nums 3"

    # Create the samples file:

    s_table = genotype_matrix.sample_table

    keep_file = osp.join(temp_dir, 'samples.keep')
    keep_table = s_table.get_individual_table()
    keep_table.to_csv(keep_file, index=False, header=False, sep="\t")

    eff_file = osp.join(temp_dir, 'variant_effect_size.txt')
    df = genotype_matrix.get_snp_table(['SNP', 'A1'])

    for i in range(betas_shape):
        df['BETA' + str(i)] = betas[:, i]

    # Remove any variants whose effect size is zero for all traits:
    df = df.loc[df[['BETA' + str(i) for i in range(betas_shape)]].sum(axis=1) != 0]

    # Standardize the genotype, if requested:
    if standardize_genotype:
        standardize_text = ' variance-standardize'
    else:
        standardize_text = ''

    df.to_csv(eff_file, index=False, sep="\t")

    output_file = osp.join(temp_dir, 'samples')

    cmd = [
        f"--bfile {genotype_matrix.bed_file}",
        f"--keep {keep_file}",
        f"--score {eff_file} 1 2 header-read cols=+scoresums{standardize_text}",
        score_col_nums,
        f"--out {output_file}",
    ]

    plink2.execute(cmd)

    if not osp.isfile(output_file + '.sscore'):
        raise FileNotFoundError

    dtypes = {'FID': str, 'IID': str}
    for i in range(betas_shape):
        dtypes.update({'PRS' + str(i): np.float64})

    chr_pgs = pd.read_csv(output_file + '.sscore',
                          delim_whitespace=True,
                          names=['FID', 'IID'] + ['PRS' + str(i) for i in range(betas_shape)],
                          skiprows=1,
                          usecols=[0, 1] + [4 + betas_shape + i for i in range(betas_shape)],
                          dtype=dtypes)
    chr_pgs = keep_table.astype({'FID': str, 'IID': str}).merge(chr_pgs)

    pgs = chr_pgs[['PRS' + str(i) for i in range(betas_shape)]].values

    if betas_shape == 1:
        pgs = pgs.flatten()

    return pgs
