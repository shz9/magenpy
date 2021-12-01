import numpy as np
import pandas as pd
import errno
import os
import subprocess
import glob
import collections
import six


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
    :return:
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


def makedir(dirs):

    if isinstance(dirs, str):
        dirs = [dirs]

    for dir in dirs:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_filenames(path, extension=None):

    if os.path.isdir(path):
        if extension == '.zarr':
            if os.path.isfile(os.path.join(path, '.zarray')):
                return [path]
            else:
                return glob.glob(os.path.join(path, '*/'))
        return glob.glob(os.path.join(path, '*'))
    else:
        if extension is None:
            return glob.glob(path + '*')
        elif extension in path:
            return [path]
        elif os.path.isfile(path + extension):
            return [path + extension]
        else:
            return glob.glob(path + '*' + extension)


def iterable(arg):
    return (
        isinstance(arg, collections.Iterable)
        and not isinstance(arg, six.string_types)
    )


def run_shell_script(cmd):

    result = subprocess.run(cmd, shell=True, capture_output=True)

    if result.stderr:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            stderr=result.stderr
        )

    return result


def delete_temp_files(prefix):
    for f in glob.glob(f"{prefix}*"):
        try:
            os.remove(f)
        except Exception as e:
            continue
