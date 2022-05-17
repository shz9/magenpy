import pandas as pd
from magenpy.utils.system_utils import get_filenames


def read_snp_filter_file(filename, snp_id_col=0):

    try:
        keep_list = pd.read_csv(filename, sep="\t", header=None).values[:, snp_id_col]
    except Exception as e:
        raise e

    return keep_list


def read_individual_filter_file(filename, iid_col=1):

    try:
        keep_list = pd.read_csv(filename, sep="\t", header=None).values[:, iid_col]
    except Exception as e:
        raise e

    return keep_list


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

    df = pd.read_csv(ldb_file_path, delim_whitespace=True)

    for chrom in df['chr'].unique():
        ld_blocks[int(chrom.replace('chr', ''))] = df.loc[df['chr'] == chrom, ['start', 'stop']].values

    return ld_blocks
