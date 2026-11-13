import numpy as np
import pandas as pd

from .chromosome_utils import normalize_chromosome

try:
    from ..stats.variant.variant_cpp import (
        is_identity_indexer_native as _is_identity_indexer_native,
    )
except (ImportError, AttributeError):
    _is_identity_indexer_native = None


def tqdm(*args, **kwargs):
    """Import tqdm only for algorithms that display iterative progress."""

    from tqdm import tqdm as _tqdm

    return _tqdm(*args, **kwargs)


def normalize_genome_build(genome_build):
    """Normalize common names for human genome assemblies."""

    if genome_build is None:
        return None

    normalized = str(genome_build).lower().replace("_", "").replace("-", "")
    normalized = normalized.replace(" ", "")
    if normalized in {"37", "b37", "build37", "grch37", "hg19", "ncbi37"}:
        return "GRCh37"
    if normalized in {"38", "b38", "build38", "grch38", "hg38", "ncbi38"}:
        return "GRCh38"
    return normalized


def validate_harmonization_builds(data_sources):
    """Reject conflicting known assemblies; allow sources with unknown builds.

    :param data_sources: Mapping from source name to a chromosome-to-data mapping.
        Objects may expose genome_build; missing or unknown builds are ignored.
    :raises ValueError: If known builds disagree within or across sources.
    """
    source_builds = {}
    for source_name, source in data_sources.items():
        for data in source.values():
            genome_build = normalize_genome_build(getattr(data, "genome_build", None))
            if genome_build is not None:
                source_builds.setdefault(source_name, set()).add(genome_build)

    genome_builds = set().union(*source_builds.values()) if source_builds else set()
    if len(genome_builds) > 1:
        build_description = ", ".join(
            f"{source}={sorted(builds)}" for source, builds in source_builds.items()
        )
        raise ValueError(
            "Cannot harmonize data sources with conflicting genome builds: "
            f"{build_description}."
        )


def match_chromosomes(
    chrom_1, chrom_2, check_patterns=("chr_", "chr:", "chr"), return_both=False
):
    """
    Given two lists of chromosome IDs, this function returns the
    chromosomes that are common to both lists. By default, the returned chromosomes
    follow the data type and order of the first list. If `return_both` is set to True,
    the function returns the common chromosomes in both lists.

    The function also accounts for common ways to encode chromosomes, such as
    chr18, chr_18, 18, etc.

    :param chrom_1: A list or numpy array of chromosome IDs
    :param chrom_2: A list or numpy array of chromosome IDs
    :param check_patterns: Additional leading prefixes to strip before standard normalization.
    :param return_both: If True, return the common chromosomes in both lists
    """

    chrom_1 = np.array(list(chrom_1))
    chrom_2 = np.array(list(chrom_2))

    def first_positions(values):
        positions = {}
        for i, value in enumerate(values):
            # Retain support for callers supplying extra prefix conventions.
            if isinstance(value, str):
                value = value.strip()
                for prefix in check_patterns:
                    if value.lower().startswith(prefix.lower()):
                        value = value[len(prefix) :]
                        break
            key = normalize_chromosome(value)
            if not pd.isna(key):
                positions.setdefault(key, i)
        return positions

    left = first_positions(chrom_1)
    right = first_positions(chrom_2)
    pairs = [(i, right[key]) for key, i in left.items() if key in right]
    if not pairs:
        return ([], []) if return_both else []
    left_positions, right_positions = np.asarray(pairs, dtype=np.intp).T
    if return_both:
        return chrom_1[left_positions], chrom_2[right_positions]
    return chrom_1[left_positions]


def is_identity_indexer(indexer, size=None, chunk_size=65536):
    """Return whether an integer indexer selects every row in its current order.

    Contiguous int32 and int64 arrays use a one-pass native implementation. The
    comparison is chunked as a fallback when the extension is unavailable or the
    input has another representation.

    :param indexer: One-dimensional positional indexer.
    :param size: Expected source size. Defaults to the indexer length.
    :param chunk_size: Maximum number of expected positions allocated per check.
    """

    indexer = np.asarray(indexer)
    if indexer.ndim != 1:
        return False
    if size is None:
        size = len(indexer)
    if len(indexer) != size:
        return False
    if size == 0:
        return True

    if (
        _is_identity_indexer_native is not None
        and indexer.flags.c_contiguous
        and indexer.dtype in (np.dtype(np.int32), np.dtype(np.int64))
    ):
        return bool(_is_identity_indexer_native(indexer, size))

    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        expected = np.arange(start, stop, dtype=np.intp)
        if not np.array_equal(indexer[start:stop], expected):
            return False
    return True


def flip_allele_statistics(table, flip, signed_statistics=("BETA", "STD_BETA", "Z")):
    """Update allele-dependent statistics in place for selected positional rows.

    Negate each available signed statistic and replace effect-allele frequency
    ``MAF`` with ``1 - MAF`` where ``flip`` is true. Other rows and columns are
    unchanged. This function neither matches variants nor changes allele labels:
    callers determine flips and assign reference A1/A2 after alignment.

    :param table: DataFrame whose rows are already aligned with the reference.
    :param flip: One-dimensional boolean mask in table row order, independent
        of the DataFrame index labels.
    :param signed_statistics: Signed statistic column names, or a single name.
    :return: The same DataFrame, updated in place without a full-table copy.
    """
    flip = np.asarray(flip, dtype=bool)
    if flip.ndim != 1 or len(flip) != len(table):
        raise ValueError("flip must be a one-dimensional mask with one value per row.")
    positions = np.flatnonzero(flip)
    if len(positions) == 0:
        return table
    if isinstance(signed_statistics, str):
        signed_statistics = [signed_statistics]

    for statistic in signed_statistics:
        if statistic in table:
            column = table.columns.get_loc(statistic)
            table.iloc[positions, column] = -table.iloc[positions, column].to_numpy()
    if "MAF" in table:
        column = table.columns.get_loc("MAF")
        table.iloc[positions, column] = 1.0 - table.iloc[positions, column].to_numpy()
    return table


def merge_snp_tables(
    ref_table,
    alt_table,
    how="inner",
    on="auto",
    signed_statistics=("BETA", "STD_BETA", "Z"),
    drop_duplicates=True,
    correct_flips=True,
    return_ref_indices=False,
    return_alt_indices=False,
):
    """
    This function takes a reference SNP table with at least 3 columns ('SNP', 'A1', `A2`)
    and matches it with an alternative table that also has these 3 columns defined. In the most recent
    implementation, we allow users to merge on any set of columns that they wish by specifying the `on`
    parameter. For example, instead of `SNP`, the user can join the SNP tables on `CHR` and `POS`, the
    chromosome number and base pair position of the SNP.

    The manner in which the join operation takes place depends on the `how` argument.
    Currently, the function supports `inner` and `left` joins.

    The function removes duplicates if `drop_duplicates` parameter is set to True

    If `correct_flips` is set to True, the function will correct summary statistics in
    the alternative table `alt_table` (e.g. BETA, MAF) based whether the A1 alleles agree between the two tables.

    :param ref_table: The reference table (pandas dataframe).
    :param alt_table: The alternative table (pandas dataframe)
    :param how: `inner` or `left`
    :param on: Which columns to use as anchors when merging. By default, we
    use `CHR` and `POS` when available in both tables, falling back to `SNP`.
    If neither definition is available, we raise a ValueError.
    :param signed_statistics: The columns with signed statistics to flip if `correct_flips` is set to True.
    :param drop_duplicates: Drop duplicate SNPs
    :param correct_flips: Correct SNP summary statistics that depend on status of alternative allele
    :param return_ref_indices: Return the indices of the remaining entries in the reference table before merging.
    :param return_alt_indices: Return the indices of the remaining entries in the alternative table before merging.
    """

    # Sanity checking steps:
    assert how in ("left", "inner")
    for tab in (ref_table, alt_table):
        assert isinstance(tab, pd.DataFrame)
        if not all([col in tab.columns for col in ("A1", "A2")]):
            raise ValueError(
                "To merge SNP tables, we require that the columns `A1` and `A2` are present."
            )

    # Determine which columns to merge on:
    if on == "auto":
        # Check that the `CHR`, `POS` columns are present in both tables:
        if all(
            [
                col in tab.columns
                for col in ("CHR", "POS")
                for tab in (ref_table, alt_table)
            ]
        ):
            on = ["CHR", "POS"]
        # Fall back to the `SNP` column:
        elif all(["SNP" in tab.columns for tab in (ref_table, alt_table)]):
            on = ["SNP"]
        else:
            raise ValueError(
                "Cannot merge SNP tables without specifying which columns to merge on."
            )
    elif isinstance(on, str):
        on = [on]

    ref_include = on + ["A1", "A2"]

    if return_ref_indices:
        ref_table = ref_table.reset_index(names="REF_IDX")
        ref_include += ["REF_IDX"]
    if return_alt_indices:
        alt_table = alt_table.reset_index(names="ALT_IDX")

    merged_table = ref_table[ref_include].merge(alt_table, how=how, on=on)

    if drop_duplicates:
        merged_table = merged_table.drop_duplicates(subset=on)

    if how == "left":
        merged_table["A1_y"] = merged_table["A1_y"].fillna(merged_table["A1_x"])
        merged_table["A2_y"] = merged_table["A2_y"].fillna(merged_table["A2_x"])

    # Assign A1 to be the one derived from the reference table:
    merged_table["A1"] = merged_table["A1_x"]
    merged_table["A2"] = merged_table["A2_x"]

    # Detect cases where the correct allele is specified in both tables:
    matching_allele = np.all(
        merged_table[["A1_x", "A2_x"]].values == merged_table[["A1_y", "A2_y"]].values,
        axis=1,
    )

    # Detect cases where the effect and reference alleles are flipped:
    flip = np.all(
        merged_table[["A2_x", "A1_x"]].values == merged_table[["A1_y", "A2_y"]].values,
        axis=1,
    )

    if how == "inner":
        # Variants to keep:
        if correct_flips:
            keep_snps = matching_allele | flip
        else:
            keep_snps = matching_allele

        # Keep only SNPs with matching alleles or SNPs with flipped alleles:
        merged_table = merged_table.take(np.flatnonzero(keep_snps))
        flip = flip[keep_snps]

    if correct_flips:
        flip_allele_statistics(merged_table, flip, signed_statistics=signed_statistics)

    merged_table = merged_table.drop(["A1_x", "A1_y", "A2_x", "A2_y"], axis=1)

    return merged_table


def build_snp_harmonization_plan(
    snp_ids,
    reference,
    alleles=None,
    allele_reference=None,
    allele_target=None,
):
    """Build one positional alignment plan for several variant data sources.

    The function operates only on arrays that have already been loaded. It finds
    the shared variants once, orders them according to ``reference``, and returns
    a positional indexer for every source. Optionally, it also removes variants
    with incompatible alleles and reports which target alleles need flipping.
    Missing identifiers and identifiers duplicated within any source are excluded.

    :param snp_ids: Mapping from source name to its one-dimensional variant ID
        array or pandas Index. Composite identifiers may be represented by a
        pandas MultiIndex.
    :param reference: Source whose SNP order defines the final canonical order.
    :param alleles: Optional mapping from source name to an ``(A1, A2)`` pair.
    :param allele_reference: Source defining the canonical allele orientation.
    :param allele_target: Source whose alleles are checked against the reference.

    :return: A dictionary with ``indexers`` and ``variant_ids``. The ``snps``
        key remains as a backwards-compatible array alias. When allele sources
        are provided, it also contains ``flip``, ``a1``, and ``a2``.
    """

    if not snp_ids:
        raise ValueError("At least one variant ID array is required for harmonization.")
    if reference not in snp_ids:
        raise KeyError(f"Reference source '{reference}' is not available.")

    source_indices = {}
    source_positions = {}

    for source, values in snp_ids.items():
        if isinstance(values, pd.MultiIndex):
            valid_positions = np.flatnonzero(
                ~values.to_frame(index=False).isna().any(axis=1).to_numpy()
            )
            index = values[valid_positions]
        else:
            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError(
                    f"Variant IDs for source '{source}' must be one-dimensional."
                )
            valid_positions = np.flatnonzero(~pd.isna(values))
            index = pd.Index(values[valid_positions])
        if not index.is_unique:
            unique = ~index.duplicated(keep=False)
            valid_positions = valid_positions[unique]
            index = index[unique]

        source_indices[source] = index
        source_positions[source] = valid_positions

    canonical_ids = source_indices[reference]
    indexers = {reference: np.arange(len(canonical_ids), dtype=np.intp)}
    keep = np.ones(len(canonical_ids), dtype=bool)

    for source, index in source_indices.items():
        if source == reference:
            continue
        positions = index.get_indexer(canonical_ids)
        keep &= positions >= 0
        indexers[source] = positions

    canonical_ids = canonical_ids[keep]
    for source in indexers:
        indexers[source] = source_positions[source][indexers[source][keep]]

    plan = {
        "indexers": indexers,
        "variant_ids": canonical_ids,
        "snps": canonical_ids.to_numpy(),
    }

    if allele_reference is None and allele_target is None:
        return plan
    if allele_reference is None or allele_target is None:
        raise ValueError("Both allele_reference and allele_target must be specified.")
    if alleles is None:
        raise ValueError("Allele arrays are required for allele harmonization.")
    for source in (allele_reference, allele_target):
        if source not in snp_ids or source not in alleles:
            raise KeyError(f"Alleles for source '{source}' are not available.")

    ref_a1_all, ref_a2_all = (np.asarray(a) for a in alleles[allele_reference])
    target_a1_all, target_a2_all = (np.asarray(a) for a in alleles[allele_target])
    if len(ref_a1_all) != len(snp_ids[allele_reference]) or len(ref_a2_all) != len(
        ref_a1_all
    ):
        raise ValueError(
            "Reference allele arrays must match the variant ID array length."
        )
    if len(target_a1_all) != len(snp_ids[allele_target]) or len(target_a2_all) != len(
        target_a1_all
    ):
        raise ValueError("Target allele arrays must match the variant ID array length.")

    ref_positions = indexers[allele_reference]
    target_positions = indexers[allele_target]
    ref_a1 = ref_a1_all[ref_positions]
    ref_a2 = ref_a2_all[ref_positions]
    target_a1 = target_a1_all[target_positions]
    target_a2 = target_a2_all[target_positions]

    matching = (ref_a1 == target_a1) & (ref_a2 == target_a2)
    flip = (ref_a1 == target_a2) & (ref_a2 == target_a1)
    compatible = matching | flip

    for source in indexers:
        indexers[source] = indexers[source][compatible]

    plan["snps"] = plan["snps"][compatible]
    plan["variant_ids"] = plan["variant_ids"][compatible]
    plan["flip"] = flip[compatible]
    plan["a1"] = ref_a1[compatible]
    plan["a2"] = ref_a2[compatible]

    return plan


def prepare_harmonization_plan(chromosome, data_sources):
    """Read chromosome metadata and plan alignment without changing sources.

    Collect table metadata and only the LD fields needed for matching. Prefer
    CHR/POS identifiers, falling back to SNP IDs consistently across sources.
    Choose variant-order and allele references separately, then delegate
    intersection and allele compatibility to build_snp_harmonization_plan.

    Return the plan and current LD rows in stored coordinates. The latter
    is None when no LD source is initialized.

    :param chromosome: Chromosome label used to construct LD CHR/POS identifiers.
    :param data_sources: Mapping of source names (genotype, sumstats, ld,
        annotation) to data objects for this chromosome. Absent sources can be
        omitted or set to None. Inputs are read without copying full tables.
    :return: Alignment plan and current LD row indices in stored coordinates.
    """
    genotype = data_sources.get("genotype")
    sumstats = data_sources.get("sumstats")
    ld = data_sources.get("ld")
    annotation = data_sources.get("annotation")
    # Collect in-memory table metadata and the alleles used for comparison.
    variant_tables = {}
    variant_ids = {}
    alleles = {}
    ld_original_positions = None
    ld_current_mask = None

    if genotype is not None:
        variant_tables["genotype"] = genotype.snp_table
        alleles["genotype"] = (genotype.a1, genotype.a2)

    if sumstats is not None:
        variant_tables["sumstats"] = sumstats.table
        if genotype is not None or ld is not None:
            if not all(column in sumstats.table.columns for column in ("A1", "A2")):
                raise ValueError(
                    "To harmonize summary statistics, both `A1` and "
                    "`A2` must be present."
                )
            alleles["sumstats"] = (sumstats.a1, sumstats.a2)

    if annotation is not None:
        variant_tables["annotation"] = annotation.table

    # Choose one identifier definition shared by every source. LD can
    # force a fallback to SNP IDs if its position metadata is unavailable.
    use_position = all(
        all(column in table.columns for column in ("CHR", "POS"))
        for table in variant_tables.values()
    )

    if ld is not None:
        stored_ids = None
        if use_position:
            try:
                stored_ids = ld.get_metadata("bp", apply_mask=False)
            except KeyError:
                use_position = False
        if not use_position:
            try:
                stored_ids = ld.get_metadata("snps", apply_mask=False)
            except KeyError as error:
                raise ValueError(
                    "LD metadata contains neither CHR/POS nor SNP IDs."
                ) from error

        ld_current_mask = ld.get_mask()
        if ld_current_mask is None:
            ld_original_positions = np.arange(len(stored_ids), dtype=np.intp)
        else:
            ld_original_positions = np.flatnonzero(ld_current_mask)

        current_ids = stored_ids[ld_original_positions]
        if use_position:
            variant_ids["ld"] = pd.MultiIndex.from_arrays(
                (
                    np.repeat(chromosome, len(current_ids)),
                    current_ids,
                ),
                names=("CHR", "POS"),
            )
        else:
            variant_ids["ld"] = current_ids

        # LD alleles are only needed when LD is the allele reference.
        if genotype is None and sumstats is not None:
            stored_a1 = ld.get_metadata("a1", apply_mask=False)
            stored_a2 = ld.get_metadata("a2", apply_mask=False)
            alleles["ld"] = (
                stored_a1[ld_original_positions],
                stored_a2[ld_original_positions],
            )

    variant_id_cols = ["CHR", "POS"] if use_position else ["SNP"]
    if not use_position and any(
        "SNP" not in table.columns for table in variant_tables.values()
    ):
        raise ValueError(
            "Cannot harmonize variants: CHR/POS are not available in "
            "every data source and at least one source has no SNP IDs."
        )

    for source, table in variant_tables.items():
        if use_position:
            variant_ids[source] = pd.MultiIndex.from_frame(table[variant_id_cols])
        else:
            variant_ids[source] = table["SNP"].to_numpy()

    # An LD matrix cannot be reordered without permuting its rows and
    # columns, so its stored order is canonical whenever it is present.
    # Otherwise retain genotype, summary-statistic, then annotation order.
    order_reference = next(
        source
        for source in ("ld", "genotype", "sumstats", "annotation")
        if source in variant_ids
    )

    allele_reference = None
    allele_target = None
    if "sumstats" in variant_ids:
        # TODO: When genotype and LD are both present, reconcile the
        # genotype A1/A2 orientation with LD as well. This requires
        # deciding how allele flips propagate to genotype dosages and
        # LD correlations; for now, as in the legacy implementation,
        # only summary statistics are allele-harmonized.
        if "genotype" in variant_ids:
            allele_reference = "genotype"
        elif "ld" in variant_ids:
            allele_reference = "ld"
        if allele_reference is not None:
            allele_target = "sumstats"

    plan = build_snp_harmonization_plan(
        variant_ids,
        reference=order_reference,
        alleles=alleles,
        allele_reference=allele_reference,
        allele_target=allele_target,
    )
    return plan, ld_original_positions


def sumstats_train_test_split(gdl, prop_train=0.9, **kwargs):
    """
    Perform a train-test split on the GWAS summary statistics data.
    This function implemented the PUMAS procedure described in

    > Zhao, Z., Yi, Y., Song, J. et al. PUMAS: fine-tuning polygenic risk scores with GWAS summary statistics.
    Genome Biol 22, 257 (2021). https://doi.org/10.1186/s13059-021-02479-9

    Specifically, the function takes harmonized LD and summary statistics data (in the form of a
    `GWADataLoader` object) and samples the marginal beta values for the training set, conditional
    on the LD matrix and the proportion of training data specified by `prop_train`.

    :param gdl: A `GWADataLoader` object containing the harmonized GWAS summary statistics and LD data.
    :param prop_train: The proportion of training data to sample.
    :param kwargs: Additional keyword arguments to pass to the
    `multivariate_normal_conditional_sampling` function.

    :return: A dictionary with the sampled beta values for the training and test sets.
    """

    # Sanity checks:
    assert 0.0 < prop_train < 1.0, (
        "The proportion of training data must be between 0 and 1."
    )
    assert gdl.sumstats_table is not None, (
        "The GWADataLoader object must have summary statistics initialized."
    )
    assert gdl.ld is not None, (
        "The GWADataLoader object must have LD matrices initialized."
    )

    from ..stats.ld.utils import multivariate_normal_conditional_sampling

    prop_test = 1.0 - prop_train

    results = {}

    for chrom in gdl.chromosomes:
        assert gdl.ld[chrom].n_snps == gdl.sumstats_table[chrom].n_snps, (
            "The number of SNPs in the LD matrix and the summary statistics table must match. Invoke the "
            "`harmonize_data` method on the GWADataLoader object to ensure that the data is harmonized."
        )

        n_per_snp = gdl.sumstats_table[chrom].n_per_snp
        n = n_per_snp.max()  # Get the GWAS sample size

        # The covariance scale is computed based on the proportion of training data:
        cov_scale = 1.0 / (prop_train * n) - (1.0 / n)
        # Use the standardized marginal beta as the mean:
        mean = gdl.sumstats_table[chrom].standardized_marginal_beta

        # Sample the training beta values:
        sampled_train_beta = multivariate_normal_conditional_sampling(
            gdl.ld[chrom], mean=mean, cov_scale=cov_scale, **kwargs
        )

        # Calculate the test beta values:
        sampled_test_beta = mean * (1.0 / prop_test) - sampled_train_beta * (
            prop_train / prop_test
        )

        # Store the results:
        results[chrom] = {
            "train_beta": sampled_train_beta,
            "train_n": n_per_snp * prop_train,
            "test_beta": sampled_test_beta,
            "test_n": n_per_snp * prop_test,
        }

    return results


def map_variants_to_genomic_blocks(
    variant_table,
    block_table,
    variant_pos_col="POS",
    block_boundary_cols=("block_start", "block_end"),
    filter_unmatched=False,
):
    """
    Merge a variant table with a genomic block table. This function assumes that the
    variant table includes a column with the positions of the variants `POS` and the block table
    contains the start and end positions of each block.

     !!! warning
         This function assumes that the tables contains data for a single chromosome only.

    :param variant_table: A pandas dataframe with variant information
    :param block_table: A pandas dataframe with block information
    :param variant_pos_col: The name of the column in the variant table that contains the positions. By default,
    this is set to `POS`.
    :param block_boundary_cols: A tuple of two strings specifying the column names in the block table that contain
    the start and end positions of the blocks. By default, this is set to `('block_start', 'block_end')`.
    :param filter_unmatched: If True, filter out variants that were not matched to a block.
    """

    # Sanity checks:
    assert variant_pos_col in variant_table.columns
    assert all([col in block_table.columns for col in block_boundary_cols])

    # Sort the variant table by position:
    variant_table = variant_table.sort_values(variant_pos_col)

    # Merge the two dataframes to assign each SNP to its corresponding block:
    merged_df = pd.merge_asof(
        variant_table,
        block_table,
        left_on=variant_pos_col,
        right_on=block_boundary_cols[0],
        direction="backward",
    )

    if filter_unmatched:
        # Filter merged_df to only include variants that were matched properly with a block:
        merged_df = merged_df.loc[
            (merged_df[variant_pos_col] >= merged_df[block_boundary_cols[0]])
            & (merged_df[variant_pos_col] < merged_df[block_boundary_cols[1]])
        ]

    return merged_df


def identify_mismatched_snps(
    gdl,
    chrom=None,
    n_iter=10,
    G=100,
    p_dentist_threshold=5e-8,
    p_gwas_threshold=1e-2,
    rsq_threshold=0.95,
    max_removed_per_iter=0.005,
):
    """
    This function implements a simple quality control procedures
    that checks that the GWAS summary statistics (Z-scores)
    are consistent with the LD reference panel. This is done
    using a simplified version of the framework outlined in the DENTIST paper:

    Improved analyses of GWAS summary statistics by reducing data heterogeneity and errors
    Chen et al. 2021

    Compared to DENTIST, the simplifications we make are:
        -   For each SNP, we sample one neighboring SNP at a time and compute the T statistic
            using that neighbor's information. The benefit of this is that we don't need to
            invert any matrices, so it's a fast operation to run.
        -   To arrive at a more robust estimate, we sample up to `k` neighbors and average
            the T-statistic across those `k` neighbors.

    NOTE: May need to re-implement this to apply some of the constraints genome-wide
    rather than on a per-chromosome basis.

    :param gdl: A `GWADataLoader` object
    :param chrom: Perform checking only on chromosome `chrom`
    :param n_iter: Number of iterations
    :param G: The number of neighboring SNPs to sample (default: 100)
    :param p_dentist_threshold: The Bonferroni-corrected P-value threshold (default: 5e-8)
    :param p_gwas_threshold: The nominal GWAS P-value threshold for partitioning variants (default: 1e-2)
    :param rsq_threshold: The R^2 threshold to select neighbors (neighbor's squared
    correlation coefficient must be less than specified threshold).
    :param max_removed_per_iter: The maximum proportion of variants removed in each iteration
    """

    # Import required modules / functions:
    from scipy import stats

    # Data preparation:
    if chrom is None:
        chromosomes = gdl.chromosomes
    else:
        chromosomes = [chrom]

    shapes = gdl.shapes
    mismatched_dict = {c: np.repeat(False, gdl.shapes[c]) for c in chromosomes}

    p_gwas_above_thres = {
        c: gdl.sumstats_table[c].p_value > p_gwas_threshold for c in chromosomes
    }
    gwas_thres_size = {c: p.sum() for c, p in p_gwas_above_thres.items()}
    converged = {c: False for c in chromosomes}

    for j in tqdm(range(n_iter), total=n_iter, desc="Identifying mismatched SNPs..."):
        for chrom in chromosomes:
            if converged[chrom]:
                continue

            ld_bounds = gdl.ld[chrom].get_masked_boundaries()
            z = gdl.z_scores[chrom]  # Obtain the z-scores
            t = np.zeros_like(z)

            # Loop over the LD matrix:
            for i, r in enumerate(gdl.ld[chrom]):
                # If the number of neighbors is less than 10, skip...
                if mismatched_dict[chrom][i] or len(r) < 10:
                    continue

                start_idx = ld_bounds[0, i]
                # Select neighbors randomly
                # Note: We are excluding neighbors whose squared correlation coefficient
                # is greater than pre-specified threshold:
                p = (np.array(r) ** 2 < rsq_threshold).astype(float)
                p /= p.sum()

                neighbor_idx = np.random.choice(len(r), p=p, size=G)
                neighbor_r = np.array(r)[neighbor_idx]

                # Predict the z-score of snp i, given the z-scores of its neighbors:
                pred_z = neighbor_r * z[start_idx + neighbor_idx]

                # Compute the Td statistic for each neighbor and average:
                t[i] = ((z[i] - pred_z) ** 2 / (1.0 - neighbor_r**2)).mean()

            # Compute the DENTIST p-value assuming a Chi-Square distribution with 1 dof.
            dentist_pval = 1.0 - stats.chi2.cdf(t, 1)
            # Use a Bonferroni correction to select mismatched SNPs:
            mismatched_snps = dentist_pval < p_dentist_threshold

            if mismatched_snps.sum() < 1:
                # If no new mismatched SNPs are identified, stop iterating...
                converged[chrom] = True
            elif j == n_iter - 1:
                # If this is the last iteration, take all identified SNPs
                mismatched_dict[chrom] = mismatched_dict[chrom] | mismatched_snps
            else:
                # Otherwise, we will perform the iterative filtering procedure
                # by splitting variants based on their GWAS p-values:

                # (1) Group S1: SNPs to remove from P_GWAS > threshold:
                mismatch_above_thres = mismatched_snps & p_gwas_above_thres[chrom]
                n_mismatch_above_thres = mismatch_above_thres.sum()
                prop_mismatch_above_thres = (
                    n_mismatch_above_thres / gwas_thres_size[chrom]
                )

                if n_mismatch_above_thres < 1:
                    # If no mismatches are detected above the threshold, filter
                    # the mismatches below the threshold and continue...
                    mismatched_dict[chrom] = mismatched_dict[chrom] | mismatched_snps
                    continue

                # Sort the DENTIST p-values by index:
                sort_d_pval_idx = np.argsort(dentist_pval)

                if prop_mismatch_above_thres > max_removed_per_iter:
                    idx_to_keep = sort_d_pval_idx[mismatch_above_thres][
                        int(gwas_thres_size[chrom] * max_removed_per_iter) :
                    ]
                    mismatch_above_thres[idx_to_keep] = False

                # (2) Group S2: SNPs to remove from P_GWAS < threshold

                # Find mismatched variants below the threshold:
                mismatch_below_thres = mismatched_snps & (~p_gwas_above_thres[chrom])
                n_mismatch_below_thres = mismatch_below_thres.sum()
                prop_mismatch_below_thres = n_mismatch_below_thres / (
                    shapes[chrom] - gwas_thres_size[chrom]
                )

                # For the mismatched variants below the threshold,
                # we remove the same proportion as the variants above the threshold:
                prop_keep_below_thres = min(
                    max_removed_per_iter, prop_mismatch_above_thres
                )

                if prop_mismatch_below_thres > prop_keep_below_thres:
                    idx_to_keep = sort_d_pval_idx[mismatch_below_thres][
                        int(
                            (shapes[chrom] - gwas_thres_size[chrom])
                            * prop_keep_below_thres
                        ) :
                    ]
                    mismatch_below_thres[idx_to_keep] = False

                # Update the number of variants above the threshold:
                gwas_thres_size[chrom] -= mismatch_above_thres.sum()

                # Update the mismatched dictionary:
                mismatched_dict[chrom] = (
                    mismatched_dict[chrom] | mismatch_below_thres | mismatch_above_thres
                )

    return mismatched_dict


def get_shared_distance_matrix(tree, tips=None):
    """
    This function takes a Biopython tree and returns the
    shared distance matrix, i.e. for a pair of clades or populations,
    time to most recent common ancestor of the pair minus the time of
    the most recent common ancestor (MRCA).
    """

    tips = tree.get_terminals() if tips is None else tips
    n_tips = len(tips)  # Number of terminal species
    sdist_matrix = np.zeros((n_tips, n_tips))  # Shared distance matrix

    for i in range(n_tips):
        for j in range(i, n_tips):
            if i == j:
                sdist_matrix[i, j] = tree.distance(tree.root, tips[i])
            else:
                mrca = tree.common_ancestor(tips[i], tips[j])
                sdist_matrix[i, j] = sdist_matrix[j, i] = tree.distance(tree.root, mrca)

    return sdist_matrix


def tree_to_rho(tree, min_corr):
    """
    This function takes a Biopython tree and a minimum correlation
    parameter and returns the correlation matrix for the effect sizes
    across populations.

    :param tree: a Biopython Phylo object
    :param min_corr: minimum correlation
    :return:
    """

    max_depth = max(tree.depths().values())
    tree.root.branch_length = min_corr * max_depth / (1.0 - min_corr)
    max_depth = max(tree.depths().values())

    for c in tree.find_clades():
        c.branch_length /= max_depth

    return tree.root.branch_length + get_shared_distance_matrix(tree)


def quantize(floats, int_dtype=np.int8):
    """
    Quantize floating point numbers to the specified integer type.
    NOTE: Assumes that the floats are in the range [-1, 1].
    :param floats: A numpy array of floats
    :param int_dtype: The integer type to quantize to.
    """

    # Infer the boundaries from the integer type
    info = np.iinfo(int_dtype)

    # NOTE: We add 1 to the info.min here to force the zero point to be exactly at 0.
    # See discussions on Scale Quantization Mapping.

    # Use as much in-place operations as possible
    # (Currently, we copy the data twice)
    scaled_floats = floats * info.max
    np.round(scaled_floats, out=scaled_floats)
    np.clip(scaled_floats, info.min + 1, info.max, out=scaled_floats)

    return scaled_floats.astype(int_dtype)


def dequantize(ints, float_dtype=np.float32):
    """
    Dequantize integers to the specified floating point type.
    NOTE: Assumes original floats are in the range [-1, 1].
    :param ints: A numpy array of integers
    :param float_dtype: The floating point data type to dequantize the integers to.
    """

    # Infer the boundaries from the integer type
    info = np.iinfo(ints.dtype)

    dq = ints.astype(float_dtype)
    dq /= info.max  # in-place multiplication

    return dq


def multinomial_rvs(n, p):
    """
    Copied from Warren Weckesser:
    https://stackoverflow.com/a/55830796

    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out
