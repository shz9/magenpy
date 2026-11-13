"""Canonical chromosome labels shared by parsers and variant data sources."""

import re

import numpy as np
import pandas as pd


_CHROMOSOME_PREFIX = re.compile(r"^chr(?::|_)?", re.IGNORECASE)
_INTEGER_LABEL = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_NAMED_CHROMOSOMES = {"X": 23, "Y": 24, "XY": 25, "M": 26, "MT": 26}
_PLINK_CHROMOSOME_DTYPE = pd.CategoricalDtype(categories=range(27), ordered=True)


def normalize_chromosome(chromosome):
    """Return a canonical chromosome key, preserving missing values.

    Strip whitespace and a leading chr, chr:, or chr_ prefix (case insensitive).
    Numeric labels become integers. Human X/Y/XY/M/MT use PLINK codes 23--26.
    Other contig names retain their spelling after prefix removal; embedded
    occurrences of 'chr' are never removed.
    """
    if pd.isna(chromosome):
        return chromosome
    if isinstance(chromosome, bytes):
        chromosome = chromosome.decode()
    label = _CHROMOSOME_PREFIX.sub("", str(chromosome).strip()).strip()
    if not label:
        return pd.NA
    named = _NAMED_CHROMOSOMES.get(label.upper())
    if named is not None:
        return named
    if _INTEGER_LABEL.fullmatch(label):
        return int(label.split(".", 1)[0])
    return label


def is_normalized_chromosome_dtype(dtype):
    """Check the canonical category schema without inspecting any row values."""
    if dtype == _PLINK_CHROMOSOME_DTYPE:
        return True
    if not isinstance(dtype, pd.CategoricalDtype) or not dtype.ordered:
        return False
    categories = dtype.categories
    if len(categories) < 27 or not categories[:27].equals(
        _PLINK_CHROMOSOME_DTYPE.categories
    ):
        return False
    extra = list(categories[27:])
    return extra == sorted(extra, key=chromosome_sort_key) and all(
        normalize_chromosome(label) == label for label in extra
    )


def normalize_chromosome_column(table, column="CHR"):
    """Store canonical chromosome labels as an ordered categorical column.

    Standard categories are PLINK IDs 0--26 (0 unknown, X=23, Y=24, XY=25,
    M/MT=26). Additional contigs follow these categories in sorted order.
    Standard chromosomes use int8 row codes; missing values have code -1.
    Category labels, rather than internal codes, identify nonstandard contigs.

    Normalize only the small category dictionary, then remap compact row codes.
    Already-canonical categorical columns are reused without allocating rows.
    The full table is never copied.
    """
    if column not in table:
        return table
    values = table[column]
    if is_normalized_chromosome_dtype(values.dtype):
        return table

    raw = pd.Categorical(values)
    labels = [normalize_chromosome(label) for label in raw.categories]
    extra = sorted(
        {label for label in labels if not pd.isna(label)} - set(range(27)),
        key=chromosome_sort_key,
    )
    dtype = (
        pd.CategoricalDtype(categories=list(range(27)) + extra, ordered=True)
        if extra
        else _PLINK_CHROMOSOME_DTYPE
    )
    # Aliases can collapse to one category, so rename_categories() is not enough.
    # Append a missing sentinel so input code -1 also maps to output code -1.
    category_codes = pd.Categorical(labels, dtype=dtype).codes
    lookup = np.append(category_codes, np.array([-1], dtype=category_codes.dtype))
    table[column] = pd.Categorical.from_codes(lookup[raw.codes], dtype=dtype)
    return table


def get_present_chromosomes(chromosomes, dropna=True):
    """Return observed chromosome labels in category order.

    Category metadata can contain unused IDs, so inspect the compact row codes
    once. Only the small unique result is filtered/sorted and decoded to labels.
    Set dropna=False when the caller also needs to detect missing chromosomes.
    Existing categorical input is reused without copying its row codes.
    """
    present = pd.Categorical(chromosomes, copy=False).unique()
    if dropna:
        present = present.dropna()
    return present.sort_values().to_numpy()


def chromosome_sort_key(chromosome):
    """Sort numeric chromosomes first, then nonstandard contig names."""
    return isinstance(chromosome, str), chromosome
