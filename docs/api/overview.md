

## Data Structures

* [GWADataLoader](GWADataLoader.md): A general class for loading multiple statistical genetics data sources and
harmonizing them for downstream analyses.
* [GenotypeMatrix](GenotypeMatrix.md): A class for representing on-disk genotype matrices. It provides 
interfaces for querying / manipulating / and performing computations on genotype data.
* [LDMatrix](LDMatrix.md): A class for representing on-disk Linkage-Disequilibrium (LD) matrices. It provides 
interfaces for querying / manipulating / and performing computations on LD data.
* [SampleTable](SampleTable.md): A class for representing data about samples (individuals), including covariates,
phenotypes, and other sample-specific metadata.
* [SumstatsTable](SumstatsTable.md): A class for representing summary statistics data from a GWAS study. It provides
interfaces for querying / manipulating / and performing computations on summary statistics data.
* [AnnotationMatrix](AnnotationMatrix.md): A class for representing variant annotations (e.g. functional annotations, 
pathogenicity scores, etc.) for a set of variants. It provides interfaces for querying / manipulating / and
performing computations on annotation data.

## Simulation

* [PhenotypeSimulator](simulation/PhenotypeSimulator.md): A general class for simulating phenotypes based on genetic data.

## Parsers

* [Sumstats Parsers](parsers/sumstats_parsers.md): A collection of parsers for reading GWAS summary statistics files in various formats.
* [Annotation Parsers](parsers/annotation_parsers.md): A collection of parsers for reading variant annotation files in various formats.
* [Plink Parsers](parsers/plink_parsers.md): A collection of parsers for reading PLINK files (BED/BIM/FAM) and other PLINK-related formats.

## Statistics

## Plotting

* [GWAS plots](plot/gwa.md): Functions for plotting various quantities / results from GWAS studies.
* [LD plots](plot/ld.md): Functions for plotting various quantities from LD matrices.

## Utilities

* [Compute utilities](utils/compute_utils.md): Utilities for computing various statistics / quantities over python data structures. 
* [Data utilities](utils/data_utils.md): Utilities for downloading and processing relevant data.
* [Executors](utils/executors.md): A collection of classes for interfacing with third party software, such as `plink`.
* [Model utilities](utils/model_utils.md): Utilities for merging / aligning / filtering GWAS data sources.
* [System utilities](utils/system_utils.md): Utilities for interfacing with the system environment (e.g. file I/O, environment variables, etc.).

## Data

