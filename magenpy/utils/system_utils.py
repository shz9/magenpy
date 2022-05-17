import errno
import os
import subprocess
import glob


def valid_url(path):

    import requests

    r = requests.head(path)
    return r.status_code == requests.codes.ok


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
    Obtain valid path names given the provided extensions.
    """

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


def run_shell_script(cmd):
    """
    Run the shell script given the command in `cmd`
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
