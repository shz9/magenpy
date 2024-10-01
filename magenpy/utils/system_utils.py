import os
import os.path as osp
import subprocess
import glob
import psutil
import sys


def available_cpu():
    """
    :return: The number of available cores on the system minus 1.
    """
    return psutil.cpu_count() - 1


def get_peak_memory_usage(include_children=False):
    """
    Get the peak memory usage of the running process in Mega Bytes (MB).

    !!! warning
        This function is only available on Unix-based systems for now.

    :param include_children: A boolean flag to include the memory usage of the child processes.
    :return: The peak memory usage of the running process in Mega Bytes (MB).
    """

    try:
        import resource
    except ImportError:
        return

    mem_usage_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if include_children:
        mem_usage_self = max(mem_usage_self, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)

    if sys.platform != 'darwin':
        mem_usage_self /= 1024
    else:
        mem_usage_self /= 1024**2

    return mem_usage_self


def get_memory_usage():
    """
    :return: The current memory usage of the running process in Mega Bytes (MB)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


def valid_url(path):
    """
    Check whether the provided `path` is a valid URL.
    :param path: A string with the URL to check.
    :return: True if the URL is valid, False otherwise.
    """

    import urllib.request

    try:
        with urllib.request.urlopen(path) as response:
            return response.getcode() == 200  # Check if the response status is OK (HTTP 200)
    except Exception:
        return False


def is_cmd_tool(name):
    """
    Check whether `name` is on PATH and marked as executable.
    From: https://stackoverflow.com/a/34177358
    :param name: A string with the name of the command-line tool.
    :return: True if the command-line tool is available, False otherwise.
    """
    from shutil import which

    return which(name) is not None


def is_path_writable(path):
    """
    Check whether the user has write-access to the provided `path`.
    This function supports checking for nested directories (i.e.,
    we iterate upwards until finding a parent directory that currently
    exists, and we check the write-access of that directory).

    :param path: A string with the path to check.

    :return: True if the path is writable, False otherwise.
    """

    # Get the absolute path first:
    path = osp.abspath(path)

    while True:

        if osp.exists(path):
            return os.access(path, os.W_OK)
        else:
            path = osp.dirname(path)
            if path == '/' or len(path) == 0:
                return False


def makedir(dirs):
    """
    Create directories on the filesystem, recursively.
    :param dirs: A string or list of strings with the paths to create.
    :raises: OSError if it fails to create the directory structure.
    """

    if isinstance(dirs, str):
        dirs = [dirs]

    for dir_l in dirs:
        try:
            os.makedirs(dir_l)
        except OSError as e:
            import errno
            if e.errno != errno.EEXIST:
                raise


def glob_s3_path(path):
    """
    Get the list of files/folders in the provided AWS S3 path. This works with wildcards.

    :param path: A string with the S3 path to list files/folders from.
    :return: A list of strings with the full paths of the files/folders.
    """

    import s3fs
    s3 = s3fs.S3FileSystem(anon=True)

    return s3.glob(path)


def get_filenames(path, extension=None):
    """
    Obtain valid and full path names given the provided `path` or prefix and extensions.

    :param path: A string with the path prefix or full path.
    :param extension: The extension for the class of files to search for.

    :return: A list of strings with the full paths of the files/folders.
    """

    if osp.isdir(path):
        if extension:
            if osp.isfile(osp.join(path, extension)):
                return [path]
            else:
                return [f for f in glob.glob(osp.join(path, '*/'))
                        if extension in f or osp.isfile(osp.join(f, extension))]
        else:
            return glob.glob(osp.join(path, '*'))
    else:
        if extension is None:
            return glob.glob(path + '*')
        elif extension in path:
            return glob.glob(path)
        elif osp.isfile(path + extension):
            return [path + extension]
        else:
            return (
                    glob.glob(osp.join(path, '*' + extension + '*')) +
                    glob.glob(osp.join(path, extension + '*')) +
                    glob.glob(path + '*' + extension + '*')
            )


def run_shell_script(cmd):
    """
    Run the shell script given the command prompt in `cmd`.
    :param cmd: A string with the shell command to run.

    :return: The result of the shell command.
    :raises: subprocess.CalledProcessError if the shell command fails.
    """

    result = subprocess.run(cmd, shell=True, capture_output=True, check=True)

    if result.stderr:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            stderr=result.stderr
        )

    return result


def delete_temp_files(prefix):
    """
    Delete temporary files with the given `prefix`.
    :param prefix: A string with the prefix of the temporary files to delete.
    """
    for f in glob.glob(f"{prefix}*"):
        try:
            os.remove(f)
        except (OSError, FileNotFoundError):
            continue
