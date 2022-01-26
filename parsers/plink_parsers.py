import pandas as pd


def parse_bim_file(plink_bfile):
    """
    From the plink documentation:
    https://www.cog-genomics.org/plink/1.9/formats#bim

        A text file with no header line, and one line per variant with the following six fields:

        - Chromosome code (either an integer, or 'X'/'Y'/'XY'/'MT'; '0' indicates unknown) or name
        - Variant identifier
        - Position in morgans or centimorgans (safe to use dummy value of '0')
        - Base-pair coordinate (1-based; limited to 231-2)
        - Allele 1 (corresponding to clear bits in .bed; usually minor)
        - Allele 2 (corresponding to set bits in .bed; usually major)

    :param plink_bfile:
    """

    if '.bim' not in plink_bfile:
        if '.bed' in plink_bfile:
            plink_bfile = plink_bfile.replace('.bed', '.bim')
        else:
            plink_bfile = plink_bfile + '.bim'

    bim_df = pd.read_csv(plink_bfile, delim_whitespace=True,
                         names=['CHR', 'SNP', 'cM', 'POS', 'A1', 'A2'])

    return bim_df


def parse_fam_file(plink_bfile):
    """
    From the plink documentation:
    https://www.cog-genomics.org/plink/1.9/formats#fam

        A text file with no header line, and one line per sample with the following six fields:

        - Family ID ('FID')
        - Within-family ID ('IID'; cannot be '0')
        - Within-family ID of father ('0' if father isn't in dataset)
        - Within-family ID of mother ('0' if mother isn't in dataset)
        - Sex code ('1' = male, '2' = female, '0' = unknown)
        - Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)

    TODO: Handle missing values here for downstream tasks

    :param plink_bfile:
    :return:
    """

    if '.bim' not in plink_bfile:
        if '.bed' in plink_bfile:
            plink_bfile = plink_bfile.replace('.bed', '.fam')
        else:
            plink_bfile = plink_bfile + '.fam'

    fam_df = pd.read_csv(plink_bfile, delim_whitespace=True,
                         names=['FID', 'IID', 'FatherID', 'MotherID', 'Sex', 'Phenotype'])

    return fam_df
