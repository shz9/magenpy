import errno
import os
import os.path as osp
import subprocess
import glob
import psutil


def available_cpu():
    return psutil.cpu_count() - 1


def valid_url(path):

    import requests

    try:
        r = requests.head(path)
        return r.status_code == requests.codes.ok
    except Exception:
        return False


def is_cmd_tool(name):
    """
    Check whether `name` is on PATH and marked as executable.
    From: https://stackoverflow.com/a/34177358
    """
    from shutil import which

    return which(name) is not None


def makedir(dirs):
    """
    Create directories on the filesystem, recursively.
    """

    if isinstance(dirs, str):
        dirs = [dirs]

    for dir in dirs:
        try:
            os.makedirs(dir)
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
    """
    for f in glob.glob(f"{prefix}*"):
        try:
            os.remove(f)
        except Exception as e:
            continue
