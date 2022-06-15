import numpy as np
import pandas as pd


def read_snp_filter_file(filename, snp_id_col=0):
    """
    Read plink-style file listing variant IDs.
    The file should not have a header and only has a single column.
    """

    try:
        keep_list = pd.read_csv(filename, sep="\t", header=None).values[:, snp_id_col]
    except Exception as e:
        raise e

    return keep_list


def read_sample_filter_file(filename):
    """
    Read plink-style file listing sample IDs.
    The file should not have a header, be tab-separated, and has two
    columns corresponding to Family ID (FID) and Individual ID (IID).
    You may also pass a file with a single-column of Individual IDs instead.
    """

    keep_list = pd.read_csv(filename, sep="\t", header=None).values

    if keep_list.shape[1] == 1:
        return keep_list[:, 0]
    elif keep_list.shape[1] == 2:
        return keep_list[:, 1]


def parse_ld_block_data(ldb_file_path):
    """
    This function takes a path to a file with the LD blocks
    and returns a dictionary with the chromosome ID and a list of the
    start and end positions for the blocks in that chromosome.
    The parser assumes that the LD block files have the ldetect format:
    https://bitbucket.org/nygcresearch/ldetect-data/src/master/

    :param ldb_file_path: The path to the LD blocks file
    """

    ld_blocks = {}

    df = pd.read_csv(ldb_file_path, delim_whitespace=True,
                     dtype={'chr': str, 'start': np.int64, 'end': np.int64})

    if df.isnull().values.any():
        raise ValueError("The LD block data contains missing information. This may result in invalid "
                         "LD boundaries. Please fix the LD block files before continuing!")

    for chrom in df['chr'].unique():
        ld_blocks[int(chrom.replace('chr', ''))] = df.loc[df['chr'] == chrom, ['start', 'stop']].values

    return ld_blocks


def parse_cluster_assignment_file(cluster_assignment_file):
    """
    Parses a file that maps each individual in the sample table to a cluster,
    and returns the pandas dataframe. The expected file should be whitespace delimited
    and contain three columns: FID, IID, and Cluster

    :param cluster_assignment_file: The path to the cluster assignment file.
    """
    try:
        clusters = pd.read_csv(cluster_assignment_file, delim_whitespace=True)
        clusters.columns = ['FID', 'IID', 'Cluster']
    except Exception as e:
        raise e

    return clusters
