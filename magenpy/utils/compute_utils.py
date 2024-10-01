import numpy as np
import pandas as pd


def generate_slice_dictionary(vec):
    """
    This utility function takes a sorted vector (e.g. numpy array),
    identifies the unique elements and generates a dictionary of slices
    delineating the start and end positions of each element in the vector.

    :param vec: A numpy array
    :return: A dictionary of slices
    """

    vals, idx = np.unique(vec, return_index=True)
    idx_sort = np.argsort(idx)

    vals = vals[idx_sort]
    idx = idx[idx_sort]

    d = {}

    for i in range(len(idx)):
        try:
            d[vals[i]] = slice(idx[i], idx[i + 1])
        except IndexError:
            d[vals[i]] = slice(idx[i], len(vec))

    return d


def intersect_arrays(arr1, arr2, return_index=False):
    """
    This utility function takes two arrays and returns the shared
    elements (intersection) between them. If return_index is set to True,
    it returns the index of shared elements in the first array.

    :param arr1: The first array
    :param arr2: The second array
    :param return_index: Return the index of shared elements in the first array

    :return: A numpy array of shared elements or their indices
    """

    # NOTE: For best and consistent results, we cast all data types to `str`
    # for now. May need a smarter solution in the future.
    common_elements = pd.DataFrame({'ID': arr1}, dtype=str).reset_index().merge(
        pd.DataFrame({'ID': arr2}, dtype=str)
    )

    if return_index:
        return common_elements['index'].values
    else:
        return common_elements['ID'].values


def generate_overlapping_windows(seq, window_size, step_size, min_window_size=1):
    """
    Generate overlapping windows of a fixed size over a sequence.

    :param seq: A numpy array of sorted values
    :param window_size: The size of each window.
    :param step_size: The step size between each window. If step_size < window_size, windows will overlap.
    :param min_window_size: The minimum size of a window. Windows smaller than this size will be discarded.

    :return: A numpy array of start and end indices of each window.
    """

    # Calculate the start of each window
    starts = np.arange(seq[0], seq[-1] - window_size, step_size)

    # Find the indices where each window starts and ends
    start_indices = np.searchsorted(seq, starts)
    end_indices = np.searchsorted(seq, starts + window_size, side='right')

    if end_indices[-1] < seq.shape[0]:
        end_indices[-1] = seq.shape[0]

    # Combine start and end indices:
    block_iter = np.column_stack((start_indices, end_indices))
    # Filter to keep only valid blocks:
    block_iter = block_iter[block_iter[:, 0] + min_window_size <= block_iter[:, 1], :]

    return block_iter


def detect_header_keywords(fname, keywords):
    """
    Detect if the first line of a file contains any of the provided keywords.
    This is used to check for whether headers are present in a file.

    :param fname: The fpath to the file
    :param keywords: A string or list of strings representing keywords to search for.

    :return: True if any of the keywords are found, False otherwise.
    """

    if isinstance(keywords, str):
        keywords = [keywords]

    with open(fname, 'r') as f:
        line = f.readline().strip()

    return any([kw in line for kw in keywords])


def is_numeric(obj):
    """
    Check if a python object is numeric. This function handles
    numpy arrays and scalars.
    :param obj: A python object
    :return: True if the object is numeric, False otherwise.
    """
    if isinstance(obj, np.ndarray):
        return np.issubdtype(obj.dtype, np.number)
    else:
        return np.issubdtype(type(obj), np.number)


def iterable(arg):
    """
    Check if an object is iterable, but not a string.
    :param arg: A python object.
    :return: True if the object is iterable, False otherwise.
    """

    import collections.abc

    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, str)
    )
