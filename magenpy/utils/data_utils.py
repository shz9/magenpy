import os.path as osp


def tgp_eur_data_path():
    """
    Return the path of the attached 1000G genotype data for
    European samples (N=378) and a subset of chromosome 22 (p=15938)
    """
    return osp.join(osp.dirname(osp.dirname(__file__)), 'data/1000G_eur_chr22')


def ukb_height_sumstats_path():
    """
    Return the path of the attached GWAS summary statistics file
    for standing height. The file contains summary statistics for
    HapMap3 variants on CHR22 and is a snapshot of the summary statistics
    published on the fastGWA database:
    https://yanglab.westlake.edu.cn/data/fastgwa_data/UKB/50.v1.1.fastGWA.gz
    """
    return osp.join(osp.dirname(osp.dirname(__file__)), 'data/ukb_height_chr22.fastGWA.gz')
