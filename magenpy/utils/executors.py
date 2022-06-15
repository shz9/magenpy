from .system_utils import is_cmd_tool, run_shell_script, available_cpu
import magenpy as mgp


class plink2Executor(object):

    def __init__(self, threads='auto'):

        if threads == 'auto':
            self.threads = available_cpu()
        else:
            self.threads = threads

        self.plink2_path = mgp.get_option('plink2_path')

        if not is_cmd_tool(self.plink2_path):
            raise Exception(f"Did not find the executable for plink2 at: {self.plink2_path}")

    def execute(self, cmd):

        cmd = [self.plink2_path] + cmd + [f'--threads {self.threads}']
        run_shell_script(" ".join(cmd))


class plink1Executor(object):

    def __init__(self, threads='auto'):

        if threads == 'auto':
            self.threads = available_cpu()
        else:
            self.threads = threads

        self.plink1_path = mgp.get_option('plink1.9_path')

        if not is_cmd_tool(self.plink1_path):
            raise Exception(f"Did not find the executable for plink at: {self.plink1_path}")

    def execute(self, cmd):

        cmd = [self.plink1_path] + cmd + [f'--threads {self.threads}']
        run_shell_script(" ".join(cmd))
