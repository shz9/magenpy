import subprocess
import sys


def _run_in_fresh_interpreter(statement):
    subprocess.run([sys.executable, "-c", statement], check=True)


def test_base_import_does_not_load_scipy_linear_algebra():
    _run_in_fresh_interpreter(
        "import sys; import magenpy; "
        "assert 'scipy.sparse.linalg' not in sys.modules; "
        "assert 'zarr' not in sys.modules; "
        "assert 'tqdm' not in sys.modules"
    )


def test_similarity_import_does_not_load_scipy_stats():
    _run_in_fresh_interpreter(
        "import sys; from magenpy.stats.ld import compute_ld_sumstats_similarity; "
        "assert 'scipy.stats' not in sys.modules"
    )
