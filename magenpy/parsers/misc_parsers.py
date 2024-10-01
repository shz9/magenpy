import numpy as np
import pandas as pd


def read_snp_filter_file(filename, snp_id_col=0):
    """
    Read plink-style file listing variant IDs.
    The file should not have a header and only has a single column.

    :param filename: The path to the file containing the SNP IDs
    :type filename: str
    :param snp_id_col: The column index containing the SNP IDs
    :type snp_id_col: int

    :return keep_list: A numpy array with the SNP IDs
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

    :param filename: The path to the file containing the sample IDs
    :type filename: str

    :return: A numpy array with the sample IDs
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

    :param ldb_file_path: The path (or URL) to the LD blocks file
    :type ldb_file_path: str

    :return: A dictionary with the chromosome ID and a list of the start
    and end positions for the blocks in that chromosome.
    """

    ld_blocks = {}

    df = pd.read_csv(ldb_file_path, sep=r'\s+')

    # Drop rows with missing values:
    df.dropna(inplace=True)
    df = df.loc[(df.start != 'None') & (df.stop != 'None')]

    # Cast the start/stop columns to integers:
    df = df.astype({'chr': str, 'start': np.int32, 'stop': np.int32})

    # Sort the dataframe:
    df = df.sort_values('start')

    for chrom in df['chr'].unique():
        ld_blocks[int(chrom.replace('chr', ''))] = df.loc[df['chr'] == chrom, ['start', 'stop']].values

    return ld_blocks


def parse_cluster_assignment_file(cluster_assignment_file):
    """
    Parses a file that maps each individual in the sample table to a cluster,
    and returns the pandas dataframe. The expected file should be whitespace delimited
    and contain three columns: FID, IID, and Cluster

    :param cluster_assignment_file: The path to the cluster assignment file.
    :type cluster_assignment_file: str

    :return: A pandas dataframe with the cluster assignments.
    """
    try:
        clusters = pd.read_csv(cluster_assignment_file, sep=r'\s+')
        clusters.columns = ['FID', 'IID', 'Cluster']
    except Exception as e:
        raise e

    return clusters
