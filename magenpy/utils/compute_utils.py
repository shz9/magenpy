import numpy as np
import pandas as pd


def generate_slice_dictionary(vec):
    """
    This utility function takes a sorted vector (e.g. numpy array),
    identifies the unique elements and generates a dictionary of slices
    delineating the start and end positions of each element in the vector.

    :param vec: A numpy array
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


def iterable(arg):
    """
    Check if an object is iterable (but not a string).
    :param arg: A python object.
    :return: True if the object is iterable, False otherwise.
    """

    import collections.abc

    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, str)
    )
