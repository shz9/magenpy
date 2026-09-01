import builtins
import subprocess
import sys

import pytest

from magenpy.utils import system_utils


def test_non_profiling_utilities_work_without_psutil(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def import_without_psutil(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            raise ModuleNotFoundError("No module named 'psutil'", name="psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_psutil)

    assert isinstance(system_utils.available_cpu(), int)
    assert system_utils.is_path_writable(tmp_path / "new_directory")


def test_system_utils_imports_without_psutil():
    script = r"""
import builtins

real_import = builtins.__import__

def import_without_psutil(name, *args, **kwargs):
    if name == "psutil" or name.startswith("psutil."):
        raise ModuleNotFoundError("No module named 'psutil'", name="psutil")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_psutil

from magenpy.utils.system_utils import available_cpu

assert isinstance(available_cpu(), int)
"""

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "profiling_utility",
    [system_utils.PeakMemoryProfiler, system_utils.get_memory_usage],
)
def test_profiling_utilities_explain_how_to_install_psutil(
    monkeypatch, profiling_utility
):
    real_import = builtins.__import__

    def import_without_psutil(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            raise ModuleNotFoundError("No module named 'psutil'", name="psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_psutil)

    with pytest.raises(ImportError, match=r"magenpy\[profiling\]"):
        if profiling_utility is system_utils.PeakMemoryProfiler:
            with profiling_utility():
                pass
        else:
            profiling_utility()


def test_profiling_utilities_with_psutil():
    pytest.importorskip("psutil")

    assert system_utils.get_memory_usage() > 0

    with system_utils.PeakMemoryProfiler(interval=0.01) as profiler:
        profiler.get_curr_memory()

    assert profiler.get_peak_memory() >= 0


@pytest.mark.parametrize(
    ("dependency", "cloud_utility", "path"),
    [
        ("s3fs", system_utils.glob_s3_path, "s3://bucket/path/*"),
        ("huggingface_hub", system_utils.glob_hf_path, "hf://repo/path/*"),
    ],
)
def test_cloud_utilities_explain_how_to_install_dependencies(
    monkeypatch, dependency, cloud_utility, path
):
    real_import = builtins.__import__

    def import_without_dependency(name, *args, **kwargs):
        if name == dependency or name.startswith(f"{dependency}."):
            raise ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)

    with pytest.raises(ImportError, match=r"magenpy\[cloud\]"):
        cloud_utility(path)
