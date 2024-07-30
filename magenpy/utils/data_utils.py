import os.path as osp


def tgp_eur_data_path():
    """
    :return: The path of the attached 1000G genotype data for
    European samples (N=378) and a subset of chromosome 22 (p=15938)
    """
    return osp.join(osp.dirname(osp.dirname(__file__)), 'data/1000G_eur_chr22')


def ukb_height_sumstats_path():
    """
    :return: The path of the attached GWAS summary statistics file
    for standing height. The file contains summary statistics for
    HapMap3 variants on CHR22 and is a snapshot of the summary statistics
    published on the fastGWA database:
    https://yanglab.westlake.edu.cn/data/fastgwa_data/UKB/50.v1.1.fastGWA.gz
    """
    return osp.join(osp.dirname(osp.dirname(__file__)), 'data/ukb_height_chr22.fastGWA.gz')


def lrld_path():
    """
    The boundaries of Long Range LD (LRLD) regions derived from here:

        https://genome.sph.umich.edu/wiki/Regions_of_high_linkage_disequilibrium_(LD)

    Which is based on the work of

    > Anderson, Carl A., et al. "Data quality control in genetic case-control association studies."
    Nature protocols 5.9 (2010): 1564-1573.

    :return: The path of the attached BED file containing long-range linkage disequilibrium
    (LD) regions in the human genome. The coordinates are in hg19/GRCh37.
    """
    return osp.join(osp.dirname(osp.dirname(__file__)), 'data/lrld_hg19_GRCh37.txt')
