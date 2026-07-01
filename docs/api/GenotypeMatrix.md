# GenotypeMatrix

`GenotypeMatrix` is the common interface used by `magenpy` for PLINK BED-backed genotype data.
Concrete implementations share the same high-level methods for sample and variant filtering,
allele-frequency summaries, scoring, GWAS, LD computation, and conversion to in-memory arrays.

`GWADataLoader` chooses the concrete class from its `backend` argument:

| Backend | Class |
| --- | --- |
| `magenpy` | `MagenpyGenotypeMatrix` |
| `bed-reader` | `bedReaderGenotypeMatrix` |
| `plink` | `plinkBEDGenotypeMatrix` |
| `xarray` | `xarrayGenotypeMatrix` |

The default backend is `magenpy`.

```python
import magenpy as mgp

gdl = mgp.GWADataLoader("path/to/data.bed")
gmat = gdl.genotype[22]
```

## Base Class

::: magenpy.GenotypeMatrix.GenotypeMatrix

## Native Backend

::: magenpy.GenotypeMatrix.MagenpyGenotypeMatrix

## bed-reader Backend

::: magenpy.GenotypeMatrix.bedReaderGenotypeMatrix

## PLINK Backend

::: magenpy.GenotypeMatrix.plinkBEDGenotypeMatrix

## xarray Backend

::: magenpy.GenotypeMatrix.xarrayGenotypeMatrix
