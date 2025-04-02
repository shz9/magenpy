import os
import os.path as osp
import subprocess
import glob
import psutil
import time
import threading


class PeakMemoryProfiler:
    """
    A context manager that monitors and tracks the peak memory usage of a process
    (and optionally its children) over a period of time. The memory usage can be
    reported in various units (bytes, MB, or GB).

    Example:

    ```
    with PeakMemoryProfiler() as profiler:
        # Code block to monitor memory usage
        ...
    ```

    Class Attributes:
    :ivar pid: The PID of the process being monitored. Defaults to the current process.
    :ivar interval: Time interval (in seconds) between memory checks. Defaults to 0.1.
    :ivar include_children: Whether memory usage from child processes is included. Defaults to True.
    :ivar unit: The unit used to report memory usage (either 'bytes', 'MB', or 'GB'). Defaults to 'MB'.
    :ivar max_memory: The peak memory usage observed during the monitoring period.
    :ivar monitoring_thread: Thread used for monitoring memory usage.
    :ivar _stop_monitoring: Event used to signal when to stop monitoring.
    """

    def __init__(self, pid=None, interval=0.1, include_children=True, unit="MB"):
        """
        Initializes the PeakMemoryProfiler instance with the provided parameters.

        :param pid: The PID of the process to monitor. Defaults to None (current process).
        :param interval: The interval (in seconds) between memory checks. Defaults to 0.1.
        :param include_children: Whether to include memory usage from child processes. Defaults to True.
        :param unit: The unit in which to report memory usage. Options are 'bytes', 'MB', or 'GB'. Defaults to 'MB'.
        """
        self.pid = pid or psutil.Process().pid  # Default to current process if no PID is provided
        self.interval = interval
        self.include_children = include_children
        self.unit = unit
        self.max_memory = 0
        self.monitoring_thread = None
        self._stop_monitoring = threading.Event()

    def __enter__(self):
        """
        Starts monitoring memory usage when entering the context block.

        :return: Returns the instance of PeakMemoryProfiler, so that we can access peak memory later.
        """
        self.process = psutil.Process(self.pid)
        self.max_memory = 0
        self._stop_monitoring.clear()  # Clear the stop flag to begin monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_memory)
        self.monitoring_thread.start()
        return self  # Return the instance so that the caller can access max_memory

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the memory monitoring when exiting the context block.

        :param exc_type: The exception type if an exception was raised in the block.
        :param exc_value: The exception instance if an exception was raised.
        :param traceback: The traceback object if an exception was raised.
        """
        self._stop_monitoring.set()  # Signal the thread to stop monitoring
        self.monitoring_thread.join()  # Wait for the monitoring thread to finish

    def get_curr_memory(self):
        """
        Get the current memory usage of the monitored process and its children.

        :return: The current memory usage in the specified unit (bytes, MB, or GB).
        :rtype: float
        """

        memory = self.process.memory_info().rss

        if self.include_children:
            # Include memory usage of child processes recursively
            for child in self.process.children(recursive=True):
                try:
                    memory += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        if self.unit == "MB":
            return memory / (1024 ** 2)  # Convert to MB
        elif self.unit == "GB":
            return memory / (1024 ** 3)  # Convert to GB
        else:
            return memory  # Return in bytes if no conversion is requested

    def _monitor_memory(self):
        """
        Monitors the memory usage of the process and its children continuously
        until the monitoring is stopped.

        This method runs in a separate thread and updates the peak memory usage
        as long as the monitoring flag is not set.
        """
        while not self._stop_monitoring.is_set():
            try:
                curr_memory = self.get_curr_memory()

                # Update max memory if a new peak is found
                self.max_memory = max(self.max_memory, curr_memory)
                time.sleep(self.interval)
            except psutil.NoSuchProcess:
                break  # Process no longer exists, stop monitoring

    def get_peak_memory(self):
        """
        Get the peak memory usage observed during the monitoring period.

        :return: The peak memory usage in the specified unit (bytes, MB, or GB).
        :rtype: float
        """
        return self.max_memory


def available_cpu():
    """
    :return: The number of available cores on the system minus 1.
    """
    return psutil.cpu_count() - 1


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
            v_url = response.getcode() == 200  # Check if the response status is OK (HTTP 200)
        return v_url
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


def setup_logger(loggers=None,
                 modules=None,
                 log_file=None,
                 log_format=None,
                 log_level='WARNING',
                 clear_file=False):
    """
    Set up the logger with the provided configuration.

    :param loggers: A list of logger instances to apply the logger configurations to.
    :param modules: A list of modules to apply the logger configurations to. This allows
    for setting up different logger configs for different modules.
    :param log_file: A string with the path to the log file.
    :param log_format: A string with the format of the log messages.
    :param log_level: A string with the logging level.
    :param clear_file: A boolean flag to clear the log file before writing.
    """

    # ------------------ Sanity checks ------------------
    assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], \
        f"Invalid log level: {log_level}"

    assert loggers is not None or modules is not None, \
        "Either `loggers` or `modules` must be provided!"

    loggers = loggers or []

    # ------------------ Set up the configurations ------------------

    import logging

    log_level = logging.getLevelName(log_level)
    handlers = []

    # Create a file handler:
    if log_file is not None:
        assert is_path_writable(log_file), f"Cannot write to {log_file}!"
        file_handler = logging.FileHandler(log_file, mode='w' if clear_file else 'a')
        file_handler.setLevel(logging.getLevelName(log_level))
        handlers.append(file_handler)

    # Create a console handler:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.getLevelName(log_level))
    handlers.append(console_handler)

    # Set up the formatter:
    log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Add the formatter to the handlers and add the handlers to the logger:
    for handler in handlers:
        handler.setFormatter(formatter)

        for lgr in loggers:
            lgr.setLevel(log_level)
            lgr.addHandler(handler)

        if modules is not None:
            for lgr_name, lgr in logging.root.manager.loggerDict.items():
                if any([m in lgr_name for m in modules]) and isinstance(lgr, logging.Logger):
                    lgr.setLevel(log_level)
                    lgr.addHandler(handler)
