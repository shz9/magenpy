from .system_utils import is_cmd_tool, run_shell_script
from magenpy import get_option


class plink2Executor(object):
    """
    A wrapper class for interfacing with the `plink2` command line tool.
    """

    def __init__(self, threads=None, verbose=True):
        """
        Initialize the plink2 executor
        :param threads: The number of threads to use for computations. If None, the number of
        threads will be set to 1.
        :type threads: int or str
        :param verbose: Whether to print the output of the command
        :type verbose: bool
        """

        self.threads = threads or 1

        self.plink2_path = get_option('plink2_path')

        if not is_cmd_tool(self.plink2_path):
            raise Exception(f"Did not find the executable for plink2 at: {self.plink2_path}")

        self.verbose = verbose

    def execute(self, cmd):
        """
        Execute a `plink2` command
        :param cmd: The flags to pass to plink2. For example, ['--bfile', 'file', '--out', 'output']
        :type cmd: list of strings
        :raises CalledProcessError: If the command returns a non-zero exit code
        """

        cmd = [self.plink2_path] + cmd + [f'--threads {self.threads}']

        from subprocess import CalledProcessError

        try:
            run_shell_script(" ".join(cmd))
        except CalledProcessError as e:

            if self.verbose:
                print("Invocation of plink2 returned the following error message:")
                print(e.stderr.decode())

            raise e


class plink1Executor(object):
    """
    A wrapper class for interfacing with the `plink1.9` command line tool.
    """

    def __init__(self, threads=None, verbose=True):
        """
        Initialize the plink1.9 executor
        :param threads: The number of threads to use for computations. If None, the number of
        threads will be set to 1.
        :type threads: int or str
        :param verbose: Whether to print the output of the command
        :type verbose: bool
        """

        self.threads = threads or 1

        self.plink1_path = get_option('plink1.9_path')

        if not is_cmd_tool(self.plink1_path):
            raise Exception(f"Did not find the executable for plink1.9 at: {self.plink1_path}")

        self.verbose = verbose

    def execute(self, cmd):
        """
        Execute a plink command
        :param cmd: The flags to pass to plink. For example, ['--bfile', 'file', '--out', 'output']
        :type cmd: list of strings
        :raises: CalledProcessError if the command fails
        """

        cmd = [self.plink1_path] + cmd + [f'--threads {self.threads}']

        from subprocess import CalledProcessError

        try:
            run_shell_script(" ".join(cmd))
        except CalledProcessError as e:
            if self.verbose:
                print("Invocation of plink returned the following error message:")
                print(e.stderr.decode())
            raise e
