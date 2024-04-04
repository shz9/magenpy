import errno
import os
import os.path as osp
import subprocess
import glob
import psutil


def available_cpu():
    """
    Get the number of available CPUs on the system.
    """
    return psutil.cpu_count() - 1


def get_memory_usage():
    """
    Get the memory usage of the current process in Mega Bytes (MB)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


def valid_url(path):
    """
    Check whether the provided `path` is a valid URL.
    :param path: A string with the URL to check.
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
    """

    if isinstance(dirs, str):
        dirs = [dirs]

    for dir_l in dirs:
        try:
            os.makedirs(dir_l)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_filenames(path, extension=None):
    """
    Obtain valid and full path names given the provided `path` or prefix and extensions.

    :param path: A string with the path prefix or full path.
    :param extension: The extension for the class of files to search for.
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
    """

    result = subprocess.run(cmd, shell=True, capture_output=True)

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
        except Exception as e:
            continue
