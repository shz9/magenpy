"""Compare GWAS summary statistics with candidate LD reference panels.

The public entry point, :func:`compute_ld_sumstats_similarity`, supports two
input modes:

* regional or chromosome-wise comparison of one summary-statistics table with
  several :class:`~magenpy.LDMatrix.LDMatrix` objects; and
* genome-wide comparison of several harmonized
  :class:`~magenpy.GWADataLoader.GWADataLoader` objects.

Both modes follow the same pipeline: harmonize alleles and variants, select a
small shared set of genomic blocks, score each candidate using allele
frequencies or an LD-based Gaussian likelihood, and convert the scores into
relative probabilities with a softmax transformation.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ...LDMatrix import LDMatrix
from ...SumstatsTable import SumstatsTable
from ...utils.compute_utils import intersect_multiple_arrays
from ...utils.model_utils import match_chromosomes, merge_snp_tables


def _normalise_candidates(candidates, labels, candidate_type, name):
    """Convert candidate inputs and their labels to validated parallel lists.

    A mapping is treated as ``label -> candidate``. For any other iterable,
    labels are either supplied separately or generated as ``candidate_0``,
    ``candidate_1``, and so on.

    :param candidates: Mapping or iterable containing candidate objects.
    :param labels: Optional labels for candidates supplied as an iterable.
    :param candidate_type: Required class for every candidate object.
    :param name: Human-readable candidate type used in error messages.
    :return: A tuple ``(candidates, labels)`` containing validated lists in
        corresponding order.
    :raises ValueError: If inputs are empty, labels are duplicated, or the
        number of labels does not match the number of candidates.
    :raises TypeError: If a candidate has the wrong type.
    """

    # Mappings provide their own labels; other iterables need explicit or
    # generated labels.
    if isinstance(candidates, Mapping):
        if labels is not None:
            raise ValueError(f"Do not pass `labels` with a mapping of {name}.")
        labels, candidates = list(candidates), list(candidates.values())
    else:
        candidates = list(candidates)
        labels = (
            [f"candidate_{i}" for i in range(len(candidates))]
            if labels is None
            else list(labels)
        )

    if not candidates or len(labels) != len(candidates):
        raise ValueError(f"Provide at least one label per {name}.")
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be unique.")
    if not all(isinstance(candidate, candidate_type) for candidate in candidates):
        raise TypeError(f"Every candidate must be {candidate_type.__name__}.")
    return candidates, labels


def _prepare_inputs(sumstats, ld_matrices, labels):
    """Validate and normalize regional or chromosome-wise inputs.

    The function copies summary statistics so that deriving Z scores and
    normalizing chromosome labels never mutates the caller's object. If a
    :class:`SumstatsTable` lacks an explicit ``Z`` column, its standard
    ``z_score`` property is used to derive one when possible.

    :param sumstats: A SumstatsTable or pandas DataFrame.
    :param ld_matrices: Candidate LDMatrix objects, as an iterable or a mapping
        from labels to matrices.
    :param labels: Optional labels when ``ld_matrices`` is an iterable.
    :return: A tuple ``(sumstats_table, ld_matrices, labels)``. The summary
        statistics are returned as a copied DataFrame.
    :raises TypeError: If summary statistics or LD candidates have invalid
        types.
    :raises ValueError: If candidate LD matrices refer to different
        chromosomes.
    """

    # Convert summary statistics to a private DataFrame copy and derive Z when
    # the SumstatsTable interface has enough information to do so.
    if isinstance(sumstats, SumstatsTable):
        table = sumstats.table.copy()
        if "Z" not in table:
            try:
                table["Z"] = SumstatsTable(table.copy()).z_score
            except KeyError:
                pass
    elif isinstance(sumstats, pd.DataFrame):
        table = sumstats.copy()
    else:
        raise TypeError("`sumstats` must be a SumstatsTable or pandas DataFrame.")

    matrices, labels = _normalise_candidates(
        ld_matrices, labels, LDMatrix, "LD matrix"
    )

    # A regional comparison is meaningful only when every candidate describes
    # the same chromosome. Missing chromosome metadata is tolerated because
    # variant matching can still proceed by SNP ID.
    chromosomes = []
    for ldm in matrices:
        try:
            chromosomes.append(ldm.chromosome)
        except KeyError:
            pass
    if chromosomes and any(
        len(match_chromosomes([chromosomes[0]], [chromosome])) == 0
        for chromosome in chromosomes[1:]
    ):
        raise ValueError("All candidate LD matrices must describe the same chromosome.")

    # Restrict multi-chromosome summary statistics to the candidate chromosome.
    # Assign the LD encoding afterwards (for example, 1 instead of "chr1") so
    # position-based matching uses identical chromosome keys.
    if chromosomes and "CHR" in table:
        sumstats_chrom, _ = match_chromosomes(
            table["CHR"].unique(), [chromosomes[0]], return_both=True
        )
        table = table.loc[table["CHR"].isin(sumstats_chrom)].copy()
        if len(sumstats_chrom):
            table["CHR"] = chromosomes[0]

    return table, matrices, labels


def _prepare_data_loaders(data_loaders, labels):
    """Extract chromosome-specific inputs from harmonized data loaders.

    Only chromosomes represented by both LD and summary statistics in every
    loader are retained. Chromosome labels are matched with
    :func:`match_chromosomes`, so common encodings such as ``1`` and ``chr1``
    are treated as equivalent. For each retained chromosome, summary
    statistics from the first loader are harmonized against every candidate LD
    matrix; variants absent from any candidate are removed later by
    :func:`_align_sumstats`.

    :param data_loaders: Candidate GWADataLoader objects, as an iterable or a
        mapping from labels to loaders. Each loader must contain ``ld`` and
        ``sumstats_table`` dictionaries that have already been harmonized
        internally.
    :param labels: Optional labels when ``data_loaders`` is an iterable.
    :return: A tuple ``(regions, matrix_regions, labels)``. ``regions`` contains
        one list of aligned candidate DataFrames per chromosome, while
        ``matrix_regions`` contains the corresponding LDMatrix objects.
    :raises ValueError: If a loader lacks LD or summary statistics, or no
        chromosome is common to all loaders.
    """

    # Import lazily to avoid a circular module import: GWADataLoader itself
    # depends on LD-related modules.
    from ...GWADataLoader import GWADataLoader

    loaders, labels = _normalise_candidates(
        data_loaders, labels, GWADataLoader, "GWADataLoader"
    )
    if any(gdl.ld is None or gdl.sumstats_table is None for gdl in loaders):
        raise ValueError(
            "Every GWADataLoader must contain harmonized LD and summary statistics."
        )

    # Use chromosome keys from the first loader as canonical keys, while
    # retaining the equivalent key used by every other loader.
    chromosome_keys = []
    reference_chromosomes = [
        chromosome
        for chromosome in loaders[0].chromosomes
        if chromosome in loaders[0].ld and chromosome in loaders[0].sumstats_table
    ]
    for chromosome in reference_chromosomes:
        keys = [chromosome]
        for gdl in loaders[1:]:
            available = set(gdl.ld).intersection(gdl.sumstats_table)
            _, matched = match_chromosomes(
                [chromosome], available, return_both=True
            )
            if not len(matched):
                break
            keys.append(matched[0])
        if len(keys) == len(loaders):
            chromosome_keys.append(keys)

    if not chromosome_keys:
        raise ValueError("The GWADataLoader objects have no chromosomes in common.")

    # Build a separate aligned region for each chromosome. Blocks are selected
    # only after all regions have been assembled, allowing one genome-wide
    # n_blocks budget to be distributed across chromosomes.
    regions, matrix_regions = [], []
    for keys in chromosome_keys:
        matrices = [gdl.ld[key] for gdl, key in zip(loaders, keys)]
        table, matrices, _ = _prepare_inputs(
            loaders[0].sumstats_table[keys[0]], matrices, labels
        )
        regions.append(_align_sumstats(table, matrices))
        matrix_regions.append(matrices)

    return regions, matrix_regions, labels


def _align_sumstats(sumstats, ld_matrices):
    """Harmonize summary statistics to LD panels and retain common variants.

    :func:`merge_snp_tables` performs identifier matching, allele-flip
    correction for signed statistics and MAF, and duplicate removal. Each
    returned DataFrame uses the allele coding of its corresponding LD matrix,
    but all frames contain the same summary-statistics variants in the same
    physical order.

    Two bookkeeping columns are added:

    * ``LD_INDEX`` is the variant's index in the underlying, unmasked LD store;
      it permits efficient block retrieval without changing an LDMatrix mask.
    * ``LD_POS`` is the base-pair position used to sort shared variants.

    ``LD_MAF`` is also added when allele frequencies are stored with the LD
    matrix.

    :param sumstats: Summary-statistics DataFrame containing ``A1`` and ``A2``
        plus either ``SNP`` or ``CHR``/``POS`` identifiers.
    :param ld_matrices: Candidate LDMatrix objects for one genomic region.
    :return: One aligned DataFrame per LD matrix. Rows correspond exactly
        across returned frames.
    :raises ValueError: If summary statistics lack allele or variant identifier
        columns.
    """

    if not {"A1", "A2"}.issubset(sumstats):
        raise ValueError("Summary statistics must contain A1 and A2.")

    # Prepare the smallest table needed by either scoring method once. The old
    # implementation copied the complete summary-statistics table for every LD
    # candidate, even though downstream scoring only consumes MAF and Z. Keep
    # the original row index as ALT_IDX so candidate intersections retain the
    # same semantics as merge_snp_tables(return_alt_indices=True).
    if "SNP" in sumstats:
        identifier_columns = ["SNP"]
    elif {"CHR", "POS"}.issubset(sumstats):
        identifier_columns = ["CHR", "POS"]
    else:
        raise ValueError(
            "Summary statistics must contain SNP or both CHR and POS identifiers."
        )
    score_columns = [column for column in ("MAF", "Z") if column in sumstats]
    merge_columns = identifier_columns + ["A1", "A2"] + score_columns
    indexed_sumstats = sumstats.loc[:, merge_columns].reset_index(names="ALT_IDX")

    aligned = []
    for ldm in ld_matrices:
        # use_original_index preserves indices into the stored matrix even when
        # the LDMatrix currently has a variant mask.
        reference = ldm.to_snp_table(
            identifier_columns + ["A1", "A2"], use_original_index=True
        )
        matched = merge_snp_tables(
            reference,
            indexed_sumstats,
            how="inner",
            correct_flips=True,
            return_ref_indices=True,
        ).rename(columns={"REF_IDX": "LD_INDEX"})

        # Attach LD metadata by stored index. Summary-statistics MAF and signed
        # statistics in `matched` have already been oriented to this LD panel.
        index = matched["LD_INDEX"].to_numpy(np.int64)
        matched["LD_POS"] = ldm.get_metadata("bp", apply_mask=False)[index]
        try:
            matched["LD_MAF"] = ldm.get_metadata("maf", apply_mask=False)[index]
        except KeyError:
            pass
        aligned.append(matched)

    # ALT_IDX identifies rows in the original summary-statistics table. Its
    # intersection guarantees that every candidate is evaluated on the same
    # variants.
    common = intersect_multiple_arrays(
        [frame["ALT_IDX"].to_numpy() for frame in aligned]
    )
    aligned = [frame.set_index("ALT_IDX").loc[common].reset_index() for frame in aligned]

    # All downstream block indices refer to this shared physical ordering.
    if len(common):
        order = np.argsort(aligned[0]["LD_POS"].to_numpy(), kind="stable")
        aligned = [frame.iloc[order].reset_index(drop=True) for frame in aligned]
    return aligned


def _select_blocks(n_variants, block_size, n_blocks):
    """Select evenly distributed, non-overlapping blocks of variant indices.

    Variants must already be sorted by genomic position. The full sequence is
    divided into up to ``n_blocks`` segments. If a segment contains more than
    ``block_size`` variants, its centered ``block_size`` variants are retained.
    Centering avoids consistently favoring the left edge of each segment.

    :param n_variants: Number of ordered, usable variants in the region.
    :param block_size: Maximum number of variants retained per block.
    :param n_blocks: Maximum number of blocks to return.
    :return: A list of integer index arrays. Blocks do not overlap and each has
        at most ``block_size`` entries.
    :raises ValueError: If ``block_size < 2`` or ``n_blocks < 1``.
    """

    if block_size < 2 or n_blocks < 1:
        raise ValueError("`block_size` must be at least 2 and `n_blocks` at least 1.")

    # Limit the count so non-empty blocks normally contain at least two
    # variants, which is the minimum useful size for an LD comparison.
    count = min(int(n_blocks), max(1, n_variants // 2))
    blocks = np.array_split(np.arange(n_variants), count)
    selected = []
    for block in blocks:
        if len(block) > block_size:
            start = (len(block) - int(block_size)) // 2
            block = block[start : start + int(block_size)]
        selected.append(block)
    return selected


def _valid_variants(aligned, method):
    """Find variants with valid inputs in every candidate panel.

    Frequency scoring requires finite ``MAF`` and ``LD_MAF`` values in [0, 1]
    for every candidate. LD-likelihood scoring requires finite Z scores for
    every candidate. Requiring joint validity preserves a fair comparison.

    :param aligned: Candidate DataFrames returned by :func:`_align_sumstats`.
    :param method: ``"frequency"`` or ``"ld_likelihood"``.
    :return: A shared boolean validity mask, or ``None`` if a required column
        is unavailable in at least one candidate.
    """

    columns = ("MAF", "LD_MAF") if method == "frequency" else ("Z",)
    if not all(set(columns).issubset(frame) for frame in aligned):
        return None

    valid = np.ones(len(aligned[0]), dtype=bool)
    for frame in aligned:
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            valid &= np.isfinite(values)
            if method == "frequency":
                valid &= (values >= 0) & (values <= 1)
    return valid


def _frequency_scores(aligned, af_scale, block_size, n_blocks):
    """Score one region using allele-frequency agreement.

    For candidate ``k``, the frequency distance is

    ``RMSE_k = sqrt(mean((MAF_sumstats - MAF_LD_k) ** 2))``.

    Because :func:`_align_sumstats` orients summary-statistics MAF to each LD
    panel first, the differences always compare frequencies of the same allele.
    The similarity score is ``-0.5 * (RMSE_k / af_scale) ** 2``; therefore,
    larger scores (values closer to zero) indicate greater agreement.

    :param aligned: Candidate DataFrames for one region.
    :param af_scale: Frequency difference treated as one scoring unit.
    :param block_size: Maximum variants selected per block.
    :param n_blocks: Maximum blocks selected within this region.
    :return: A tuple ``(selected_frames, scores, distances, n_blocks,
        n_variants)``. ``distances`` contains per-candidate frequency RMSE.
    :raises ValueError: If ``af_scale`` is invalid or frequency columns are
        unavailable.
    """

    if not np.isfinite(af_scale) or af_scale <= 0:
        raise ValueError("`af_scale` must be positive and finite.")
    valid = _valid_variants(aligned, "frequency")
    if valid is None:
        raise ValueError("Allele frequencies are not available for every input.")

    # Apply one shared validity mask and one shared block selection so every
    # candidate is compared using identical summary-statistics variants.
    aligned = [frame.loc[valid].reset_index(drop=True) for frame in aligned]
    blocks = _select_blocks(len(aligned[0]), block_size, n_blocks)
    selected = np.concatenate(blocks)
    aligned = [frame.iloc[selected].reset_index(drop=True) for frame in aligned]
    distances = np.array(
        [
            np.sqrt(np.mean((frame["MAF"] - frame["LD_MAF"]) ** 2))
            for frame in aligned
        ]
    )
    return (
        aligned,
        -0.5 * (distances / af_scale) ** 2,
        distances,
        len(blocks),
        len(selected),
    )


def _load_ld_block(ldm, indices, visible_indices=None):
    """Load a dense LD block for arbitrary stored variant indices.

    The smallest contiguous span containing ``indices`` is loaded through
    :meth:`LDMatrix.load_data`, which handles dequantization and symmetrization.
    The requested (possibly non-contiguous) variants are then selected from the
    span. This avoids materializing the full chromosome-wide LD matrix.

    :param ldm: Source LDMatrix.
    :param indices: Variant indices in the original on-disk LD store.
    :param visible_indices: Optional cached stored indices retained by the
        current LDMatrix mask.
    :return: A symmetric dense NumPy array ordered like ``indices``.
    """

    indices = np.asarray(indices, dtype=np.int64)
    start, end = indices.min(), indices.max() + 1
    # Most production LD stores are quantized integers or float32. Dequantize
    # those stores to float32 while the potentially much larger contiguous span
    # is sparse, then promote only the selected dense block for linear algebra.
    # Preserve float64 data when the source itself was stored at that precision.
    stored_dtype = np.dtype(ldm.stored_dtype)
    load_dtype = (
        np.float32
        if np.issubdtype(stored_dtype, np.integer) or stored_dtype.itemsize <= 4
        else np.float64
    )
    matrix = ldm.load_data(
        start_row=start,
        end_row=end,
        dtype=load_dtype,
        return_symmetric=True,
        return_as_csr=True,
    )

    # load_data respects an active LDMatrix mask. Translate stored indices to
    # row positions within the masked span before taking the final subset.
    if ldm.is_mask_set:
        visible = (
            np.flatnonzero(ldm.get_mask())
            if visible_indices is None
            else visible_indices
        )
        visible = visible[(visible >= start) & (visible < end)]
        local_indices = np.searchsorted(visible, indices)
    else:
        local_indices = indices - start
    block = matrix[local_indices][:, local_indices].toarray()
    return block.astype(np.float64, copy=False)


def _ld_likelihood_scores(
    aligned, ld_matrices, block_size, n_blocks, regularization, z_clip
):
    """Score one region using the likelihood of its GWAS Z scores.

    For every selected block and candidate panel, this function evaluates

    ``Z_block ~ MVN(0, (1 - regularization) * R + regularization * I)``,

    where ``R`` is the candidate LD block. Regularization improves numerical
    stability and reduces sensitivity to reference-panel sampling noise. Block
    log-likelihoods are summed and divided by the total number of selected
    variants, making scores comparable when fewer than the requested number of
    variants is available.

    :param aligned: Candidate DataFrames for one region.
    :param ld_matrices: LDMatrix corresponding to each aligned DataFrame.
    :param block_size: Maximum variants selected per block.
    :param n_blocks: Maximum blocks selected within this region.
    :param regularization: Weight of the identity matrix in the regularized LD
        covariance; must be in ``(0, 1]``.
    :param z_clip: Optional symmetric bound applied to Z scores before scoring.
    :return: A tuple ``(filtered_frames, scores, n_blocks, n_variants)``. Scores
        are mean log-likelihood per selected variant.
    :raises ValueError: If Z scores are unavailable, regularization is invalid,
        or an LD block remains non-positive-semidefinite.
    """

    valid = _valid_variants(aligned, "ld_likelihood")
    if valid is None:
        raise ValueError("LD-likelihood scoring requires Z scores.")
    if not 0 < regularization <= 1:
        raise ValueError("`regularization` must be in (0, 1].")

    # scipy.stats is only needed by the substantially more expensive
    # likelihood path. Keep frequency-only comparisons from importing it.
    from scipy.stats import multivariate_normal

    aligned = [frame.loc[valid].reset_index(drop=True) for frame in aligned]
    if len(aligned[0]) < 2:
        return aligned, np.full(len(aligned), np.nan), 0, 0

    blocks = _select_blocks(len(aligned[0]), block_size, n_blocks)
    scores = np.zeros(len(aligned))

    # Convert pandas columns and masks once per candidate rather than once for
    # every candidate/block pair. Clipping is likewise block-independent.
    z_values = [
        frame["Z"].to_numpy(dtype=float, copy=False) for frame in aligned
    ]
    if z_clip is not None:
        z_values = [np.clip(values, -z_clip, z_clip) for values in z_values]
    ld_indices = [
        frame["LD_INDEX"].to_numpy(dtype=np.int64, copy=False)
        for frame in aligned
    ]
    visible_indices = [
        np.flatnonzero(ldm.get_mask()) if ldm.is_mask_set else None
        for ldm in ld_matrices
    ]

    # Summary statistics have been separately oriented to each LD candidate,
    # so frame-specific Z scores must be paired with the corresponding matrix.
    for block in blocks:
        for i, ldm in enumerate(ld_matrices):
            z = z_values[i][block]
            covariance = _load_ld_block(
                ldm, ld_indices[i][block], visible_indices[i]
            )
            covariance *= 1 - regularization
            covariance.flat[:: len(block) + 1] += regularization
            try:
                scores[i] += multivariate_normal.logpdf(
                    z, mean=np.zeros(len(block)), cov=covariance, allow_singular=True
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                raise ValueError(
                    "An LD block is not positive semidefinite; increase `regularization`."
                ) from error

    n_used = sum(map(len, blocks))
    return aligned, scores / n_used, len(blocks), n_used


def _allocate_blocks(regions, method, n_blocks):
    """Allocate a genome-wide block budget across chromosome regions.

    The method-specific usable-variant counts are concatenated conceptually in
    chromosome order. ``n_blocks`` evenly spaced midpoint targets are placed on
    that genome-wide axis, and each target is assigned to its containing
    chromosome. Consequently, chromosomes receive blocks approximately in
    proportion to their usable variant counts, and no block can cross a
    chromosome boundary.

    :param regions: Aligned candidate DataFrames grouped by chromosome.
    :param method: ``"frequency"`` or ``"ld_likelihood"``.
    :param n_blocks: Total block budget across all regions.
    :return: Integer array containing the number of blocks allocated to each
        region. Its sum is ``n_blocks`` when usable variants are available,
        although the eventual number of non-empty blocks can be smaller for
        very small regions.
    """

    # Chromosomes lacking inputs for the selected method receive zero weight.
    counts = np.array(
        [
            0 if (valid := _valid_variants(region, method)) is None else valid.sum()
            for region in regions
        ]
    )
    if not counts.sum():
        return np.zeros(len(regions), dtype=int)

    # Midpoint targets avoid systematically selecting either end of the
    # cumulative genome-wide variant sequence.
    targets = (np.arange(n_blocks) + 0.5) * counts.sum() / n_blocks
    allocation = np.bincount(
        np.searchsorted(np.cumsum(counts), targets, side="right"),
        minlength=len(regions),
    )
    return allocation


def _score_regions(
    regions,
    matrix_regions,
    method,
    af_scale,
    block_size,
    n_blocks,
    regularization,
    z_clip,
):
    """Score and aggregate one or more chromosome regions.

    This is the common scoring path for both public input modes. A regional
    request supplies one element in ``regions``; a genome-wide request supplies
    one per common chromosome.

    Regional frequency RMSE values are combined by accumulating squared error
    weighted by the number of selected variants, then taking a genome-wide
    square root. Regional LD scores are mean log-likelihoods, so they are
    converted back to log-likelihood sums before aggregation and divided by the
    genome-wide selected-variant count at the end.

    :param regions: Sequence of chromosome regions. Each region is a list of
        aligned candidate DataFrames.
    :param matrix_regions: Sequence matching ``regions`` that contains the
        candidate LDMatrix objects for each chromosome.
    :param method: ``"frequency"`` or ``"ld_likelihood"``.
    :param af_scale: Frequency-score scale passed to :func:`_frequency_scores`.
    :param block_size: Maximum variants selected per block.
    :param n_blocks: Total block budget across all regions.
    :param regularization: LD covariance regularization weight.
    :param z_clip: Optional Z-score clipping threshold.
    :return: A tuple ``(scores, frequency_rmse, n_blocks, n_variants)``.
        ``frequency_rmse`` is NaN for LD-likelihood scoring.
    """

    allocation = _allocate_blocks(regions, method, n_blocks)
    n_candidates = len(regions[0])
    aggregate = np.zeros(n_candidates)
    total_variants = total_blocks = 0

    for aligned, matrices, region_blocks in zip(
        regions, matrix_regions, allocation
    ):
        if region_blocks == 0:
            continue
        if method == "frequency":
            _, _, values, blocks_used, n_used = _frequency_scores(
                aligned, af_scale, block_size, region_blocks
            )
            # Convert RMSE back to a sum of squared errors so chromosome-level
            # results can be combined without giving small regions equal weight.
            aggregate += values**2 * n_used
        else:
            _, values, blocks_used, n_used = _ld_likelihood_scores(
                aligned,
                matrices,
                block_size,
                region_blocks,
                regularization,
                z_clip,
            )
            # Convert mean per-variant log-likelihood back to a sum before
            # combining chromosomes.
            aggregate += values * n_used
        total_variants += n_used
        total_blocks += blocks_used

    # Convert accumulated values back to the public score scale.
    if method == "frequency" and total_variants:
        aggregate = np.sqrt(aggregate / total_variants)
        return -0.5 * (aggregate / af_scale) ** 2, aggregate, total_blocks, total_variants

    scores = aggregate / total_variants if total_variants else aggregate
    return scores, np.full(n_candidates, np.nan), total_blocks, total_variants


def compute_ld_sumstats_similarity(
    sumstats=None,
    ld_matrices=None,
    labels=None,
    method="auto",
    *,
    data_loaders=None,
    priors=None,
    temperature=1.0,
    af_scale=0.05,
    block_size=150,
    n_blocks=5,
    min_variants=20,
    regularization=0.05,
    z_clip=8.0,
):
    """Measure relative agreement between GWAS summary statistics and LD panels.

    There are two mutually exclusive input modes:

    1. **Regional/chromosome-wise:** Pass one ``sumstats`` object and candidate
       ``ld_matrices``. Every LD matrix must describe the same region or
       chromosome.
    2. **Genome-wide:** Pass candidate ``data_loaders``. Each GWADataLoader must
       contain LD and summary statistics that have already been harmonized
       internally. Only chromosomes available in every loader are compared.

    Genome-wide blocks are allocated across chromosomes approximately in
    proportion to the number of usable variants and never span chromosome
    boundaries. In either mode, candidates are compared using exactly the same
    summary-statistics variants.

    **Scoring methods**

    ``method="frequency"`` compares allele-aligned A1 frequencies using RMSE.
    The returned score is ``-0.5 * (RMSE / af_scale) ** 2``.

    ``method="ld_likelihood"`` treats the Z scores in each selected block as
    multivariate normal with covariance given by the regularized candidate LD
    matrix. Its score is the mean log-likelihood per selected variant.

    ``method="auto"`` first attempts frequency scoring. It falls back to LD
    likelihood when fewer than ``min_variants`` frequency-complete variants
    can be selected.

    Both methods select up to ``n_blocks`` blocks with at most ``block_size``
    variants each. ``min_variants`` applies to the total selected across all
    blocks and, in genome-wide mode, across all chromosomes.

    Candidate probabilities are calculated as
    ``softmax(score / temperature + log(prior))``. They express relative
    evidence only among the supplied candidates; they are not calibrated
    ancestry posterior probabilities. In particular, omitting the true or a
    closely related reference population can make the largest probability
    misleadingly high.

    :param sumstats: Regional/chromosome summary statistics as a SumstatsTable
        or pandas DataFrame. ``A1`` and ``A2`` are required. Frequency scoring
        also requires ``MAF``; likelihood scoring requires ``Z``, or ``BETA``
        and ``SE`` from which a SumstatsTable can derive Z.
    :param ld_matrices: Candidate regional/chromosome LDMatrix objects. May be
        an iterable or a ``label -> LDMatrix`` mapping.
    :param labels: Optional unique candidate labels when ``ld_matrices`` or
        ``data_loaders`` is an iterable. Do not pass labels with a mapping.
    :param method: Scoring method: ``"auto"``, ``"frequency"``, or
        ``"ld_likelihood"``.
    :param data_loaders: Harmonized GWADataLoader candidates for genome-wide
        scoring. May be an iterable or a ``label -> GWADataLoader`` mapping.
        Do not also pass ``sumstats`` or ``ld_matrices``.
    :param priors: Optional positive prior weights. Supply one per candidate in
        candidate order, or a mapping keyed by candidate label. Uniform priors
        are used by default.
    :param temperature: Positive softmax temperature. Values above one produce
        less concentrated relative probabilities; values below one produce
        more concentrated probabilities.
    :param af_scale: Allele-frequency RMSE corresponding to one unit in the
        frequency score. Smaller values penalize discrepancies more strongly.
    :param block_size: Maximum number of variants selected per block. Must be at
        least two.
    :param n_blocks: Maximum number of blocks across the entire comparison.
        This is a genome-wide total in data-loader mode, not a per-chromosome
        value.
    :param min_variants: Minimum total variants selected across all blocks.
    :param regularization: Identity-matrix weight in the LD covariance
        ``(1 - regularization) * R + regularization * I``. Must be in ``(0, 1]``.
    :param z_clip: Optional positive symmetric clipping threshold for Z scores.
        Set to ``None`` to disable clipping.
    :return: A DataFrame indexed by candidate label and sorted by decreasing
        ``probability``. Columns are ``method``, ``score``, ``probability``,
        ``frequency_rmse`` (NaN for likelihood scoring), ``n_variants`` (total
        selected variants), and ``n_blocks`` (actual selected blocks).
    :raises ValueError: If input modes are mixed, parameters are invalid, too
        few usable variants are found, or scores cannot be computed.

    Examples
    --------
    Regional or chromosome-wise comparison::

        result = compute_ld_sumstats_similarity(
            sumstats,
            {"AFR": afr_ld, "EUR": eur_ld},
        )

    Genome-wide comparison::

        result = compute_ld_sumstats_similarity(
            data_loaders={"AFR": afr_gdl, "EUR": eur_gdl},
            n_blocks=5,
            block_size=150,
        )
    """

    # Validate scoring and sampling parameters before inspecting potentially
    # large input objects.
    if method not in {"auto", "frequency", "ld_likelihood"}:
        raise ValueError("Unknown similarity method.")
    if min_variants < 2 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("`min_variants` must be at least 2 and `temperature` positive.")
    if block_size < 2 or n_blocks < 1 or min_variants > block_size * n_blocks:
        raise ValueError(
            "Require `block_size >= 2`, `n_blocks >= 1`, and "
            "`min_variants <= block_size * n_blocks`."
        )
    if z_clip is not None and (not np.isfinite(z_clip) or z_clip <= 0):
        raise ValueError("`z_clip` must be positive and finite.")

    # Normalize either input mode to the same representation: a list of
    # regions, each containing aligned frames and corresponding LD matrices.
    if data_loaders is not None:
        if sumstats is not None or ld_matrices is not None:
            raise ValueError(
                "Pass either `data_loaders` or `sumstats` with `ld_matrices`, not both."
            )
        regions, matrix_regions, labels = _prepare_data_loaders(
            data_loaders, labels
        )
    else:
        if sumstats is None or ld_matrices is None:
            raise ValueError(
                "Pass `sumstats` and `ld_matrices`, or use `data_loaders`."
            )
        sumstats, matrices, labels = _prepare_inputs(
            sumstats, ld_matrices, labels
        )
        regions = [_align_sumstats(sumstats, matrices)]
        matrix_regions = [matrices]

    # Bundle shared arguments because auto mode may attempt frequency scoring
    # before falling back to LD likelihood.
    score_kwargs = {
        "regions": regions,
        "matrix_regions": matrix_regions,
        "af_scale": af_scale,
        "block_size": block_size,
        "n_blocks": n_blocks,
        "regularization": regularization,
        "z_clip": z_clip,
    }
    result = None
    if method == "auto":
        # Frequency scoring is substantially cheaper because it never loads LD
        # blocks. Reuse its result if enough variants are available.
        result = _score_regions(method="frequency", **score_kwargs)
        method = "frequency" if result[-1] >= min_variants else "ld_likelihood"
        if method == "ld_likelihood":
            result = None

    if result is None:
        result = _score_regions(method=method, **score_kwargs)
    scores, distances, blocks_used, n_used = result

    # Enforce the threshold after block selection, not merely after matching.
    if n_used < min_variants:
        raise ValueError(
            f"Only {n_used} aligned, usable variants were found; {min_variants} are required."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Could not calculate finite similarity scores.")

    # Convert prior specifications to candidate order. Normalization is not
    # required because softmax is invariant to a shared multiplicative factor.
    if priors is None:
        priors = np.ones(len(labels))
    elif isinstance(priors, Mapping):
        priors = np.array([priors[label] for label in labels], dtype=float)
    else:
        priors = np.asarray(priors, dtype=float)
    if (
        priors.shape != (len(labels),)
        or not np.all(np.isfinite(priors))
        or np.any(priors <= 0)
    ):
        raise ValueError("Provide one positive prior per LD matrix.")

    # Temperature-scaled scores plus log-priors form the relative softmax
    # logits. Subtracting the maximum keeps the NumPy exponential stable and
    # avoids importing scipy.special solely for this small calculation.
    logits = scores / temperature + np.log(priors)
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()

    # The result is sorted so the best-supported candidate appears first.
    result = pd.DataFrame(
        {
            "label": labels,
            "method": method,
            "score": scores,
            "probability": probabilities,
            "frequency_rmse": distances,
            "n_variants": n_used,
            "n_blocks": blocks_used,
        }
    ).set_index("label")
    return result.sort_values("probability", ascending=False)


__all__ = ["compute_ld_sumstats_similarity"]
