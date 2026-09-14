import warnings

import numpy as np
import pandas as pd

from magenpy.AnnotationMatrix import AnnotationMatrix


def test_filter_snps_handles_fragmented_table_without_warning():
    table = pd.DataFrame(
        {
            "CHR": [1, 1, 1],
            "SNP": ["rs1", "rs2", "rs3"],
            "POS": [10, 20, 30],
        }
    )

    # Repeated insertion deliberately reproduces a highly fragmented table.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for annotation_idx in range(150):
            table[f"annotation_{annotation_idx}"] = np.arange(len(table))

    annotation_matrix = AnnotationMatrix(table)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        annotation_matrix.filter_snps(extract_snps=["rs1", "rs3"])

    assert annotation_matrix.snps.tolist() == ["rs1", "rs3"]
    assert annotation_matrix.table.index.tolist() == [0, 1]
    assert "index" not in annotation_matrix.table.columns
