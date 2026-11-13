"""Resource ownership and error-path cleanup for genotype backends."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magenpy.GenotypeMatrix import plinkBEDGenotypeMatrix, xarrayGenotypeMatrix


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize(
    "method,module,function,args,column",
    [
        ("score", "magenpy.stats.score.utils", "score_plink2", ([1.0],), None),
        ("perform_gwas", "magenpy.stats.gwa.utils", "perform_gwa_plink2", (), None),
        (
            "compute_allele_frequency",
            "magenpy.stats.variant.utils",
            "compute_allele_frequency_plink2",
            (),
            "MAF",
        ),
        (
            "compute_sample_size_per_snp",
            "magenpy.stats.variant.utils",
            "compute_sample_size_per_snp_plink2",
            (),
            "N",
        ),
    ],
)
def test_plink_removes_only_operation_directory_even_on_error(
    tmp_path, monkeypatch, fail, method, module, function, args, column
):
    shared_file = tmp_path / "other_matrix.txt"
    shared_file.write_text("preserve")
    matrix = plinkBEDGenotypeMatrix(
        snp_table=pd.DataFrame({"SNP": ["rs1"]}), temp_dir=str(tmp_path)
    )
    directories = []
    expected = np.array([0.25])

    def run(*args, temp_dir, **kwargs):
        directory = Path(temp_dir)
        directories.append(directory)
        (directory / "output.log").write_text("temporary output")
        if fail:
            raise RuntimeError("PLINK failed")
        return expected

    monkeypatch.setattr(import_module(module), function, run)
    if fail:
        # Retain the exception/traceback: removal must not depend on garbage
        # collection of the TemporaryDirectory after a stack frame disappears.
        with pytest.raises(RuntimeError, match="PLINK failed") as error:
            getattr(matrix, method)(*args)
        assert error.value.__traceback__ is not None
    else:
        result = getattr(matrix, method)(*args)
        if column is None:
            assert result is expected
        else:
            np.testing.assert_array_equal(matrix.snp_table[column], expected)
    assert len(directories) == 1
    assert not directories[0].exists()
    matrix.cleanup()
    matrix.cleanup()
    assert shared_file.read_text() == "preserve"


def test_xarray_cleanup_closes_attached_resource_once(tmp_path):
    xr = pytest.importorskip("xarray")
    data = xr.DataArray(np.zeros((2, 3)), dims=("sample", "variant"))
    closed = []
    data.set_close(lambda: closed.append(True))
    matrix = xarrayGenotypeMatrix(xr_mat=data, temp_dir=str(tmp_path))
    matrix.cleanup()
    matrix.cleanup()
    assert closed == [True]
    assert tmp_path.is_dir()
