import pandas as pd


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
