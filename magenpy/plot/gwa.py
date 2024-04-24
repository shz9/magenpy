from typing import Union
from ..GWADataLoader import GWADataLoader
from ..SumstatsTable import SumstatsTable
import matplotlib.pylab as plt
import pandas as pd
import numpy as np


def manhattan(input_data: Union[GWADataLoader, SumstatsTable, pd.DataFrame],
              x_col=None,
              y_col='NLOG10_PVAL',
              scatter_kwargs=None,
              highlight_snps=None,
              highlight_snps_kwargs=None,
              chrom_sep_color='#f0f0f0',
              add_bonf_line=True,
              bonf_line_kwargs=None):

    """
    Generate Manhattan plot where the x-axis is the genomic position (in BP)
    and the y-axis is the -log10(p-value) or some other statistic of the user's choice.

    :param input_data: An instance of `SumstatsTable` / `GWADataLoader` or a `pandas.DataFrame` from
    which data about the positions of the SNPs and their statistics will be extracted. If `x_col` and `y_col` are
    not provided, the function will assume that the input data contains columns named 'POS' and 'NLOG10_PVAL',
    for the position in base pairs and the -log10(p-value) respectively.
    :param x_col: The column name in the input data that contains the x-axis values. Defaults to None. If the
    user passes `GWADataLoader` or `SumstatsTable` objects, we attempt to extract the x-axis
    values from the 'POS' (position in base pairs) column.
    :param y_col: The column name in the input data that contains the y-axis values (default: 'NLOG10_PVAL').
    :param scatter_kwargs: A dictionary of keyword arguments to pass to the `plt.scatter` function.
    This can be used to customize the appearance of the points on the scatter plot.
    :param highlight_snps: A list or array of SNP rsIDs to highlight on the scatter plot.
    :param highlight_snps_kwargs: A dictionary of keyword arguments to pass to the `plt.scatter` function
    to customize the appearance of the highlighted SNPs.
    :param chrom_sep_color: The color for the chromosome separator block.
    :param add_bonf_line: If True, add a line indicating the Bonferroni significance threshold.
    :param bonf_line_kwargs: The color of the Bonferroni significance threshold line.

    """

    # -------------------------------------------------------
    # Process the input data:

    if isinstance(input_data, SumstatsTable):
        if x_col is None:
            x_col = 'POS'
        input_data = {c: ss.to_table(col_subset=('CHR', 'SNP', x_col, y_col))
                      for c, ss in input_data.split_by_chromosome().items()}
    elif isinstance(input_data, GWADataLoader):
        if x_col is None:
            x_col = 'POS'
        input_data = input_data.to_summary_statistics_table(col_subset=('CHR', 'SNP', x_col, y_col),
                                                            per_chromosome=True)
    elif isinstance(input_data, pd.DataFrame):

        # Sanity checks:
        assert 'CHR' in input_data.columns, "The input data must contain a column named 'CHR'."

        # If the x-axis column is not provided, we assume that the user wishes to plot the
        # variant rank on the x-axis.
        if x_col is not None:
            assert x_col in input_data.columns, f"The input data must contain a column named '{x_col}'."
        else:
            x_col = 'Variant order'
            input_data['Variant order'] = np.arange(1, input_data.shape[0] + 1)

        assert y_col in input_data.columns, f"The input data must contain a column named '{y_col}'."

        if highlight_snps is not None:
            assert 'SNP' in input_data.columns

        input_data = {c: ss for c, ss in input_data.groupby('CHR')}
    else:
        raise ValueError("The input data must be an instance of `SumstatsTable`, `GWADataLoader` "
                         "or a `pandas.DataFrame`.")

    # -------------------------------------------------------
    # Process the SNPs to be highlighted:

    if highlight_snps is not None:
        if isinstance(highlight_snps, list):
            highlight_snps = np.array(highlight_snps)

        highlight_snps = {c: np.in1d(p['SNP'], highlight_snps) for c, p in input_data.items()}

    # -------------------------------------------------------
    # Add custom scatter plot arguments (if not provided)
    if scatter_kwargs is None:
        scatter_kwargs = {'marker': '.', 'alpha': 0.3, 'color': '#808080'}
    else:
        # Only update the keys that are not already present in the dictionary:
        scatter_kwargs = {**scatter_kwargs, **{'marker': '.', 'alpha': 0.3, 'color': '#808080'}}

    if highlight_snps is not None:
        if highlight_snps_kwargs is None:
            highlight_snps_kwargs = {'marker': 'o', 'color': 'red', 'zorder': 2}
        else:
            # Only update the keys that are not already present in the dictionary:
            highlight_snps_kwargs = {**highlight_snps_kwargs, **{'marker': 'o', 'color': 'red', 'zorder': 2}}

    # Add custom Bonferroni line arguments (if not provided)
    if bonf_line_kwargs is None:
        bonf_line_kwargs = {'color': '#b06a7a', 'ls': '--', 'zorder': 1}
    else:
        # Only update the keys that are not already present in the dictionary:
        bonf_line_kwargs = {**bonf_line_kwargs, **{'color': '#b06a7a', 'ls': '--', 'zorder': 1}}

    # -------------------------------------------------------

    if y_col == 'NLOG10_PVAL' and add_bonf_line:
        # Add Bonferroni significance threshold line:
        plt.axhline(-np.log10(0.05 / 1e6), **bonf_line_kwargs)

    # -------------------------------------------------------
    # Iterate through chromosomes and generate scatter plots:

    starting_pos = 0
    ticks = []
    chrom_spacing = .1*min([p[x_col].max() - p[x_col].min() for c, p in input_data.items()])

    unique_chr = sorted(list(input_data.keys()))

    for i, c in enumerate(unique_chr):

        min_pos = input_data[c][x_col].min()
        max_pos = input_data[c][x_col].max()

        xmin = min_pos + starting_pos
        xmax = max_pos + starting_pos

        if i % 2 == 1:
            plt.axvspan(xmin=xmin, xmax=xmax, zorder=0, color=chrom_sep_color)

        ticks.append((xmin + xmax) / 2)

        plt.scatter(input_data[c][x_col] + starting_pos,
                    input_data[c][y_col],
                    **scatter_kwargs)

        if highlight_snps is not None:
            plt.scatter((input_data[c][x_col] + starting_pos)[highlight_snps[c]],
                        input_data[c][y_col][highlight_snps[c]],
                        **highlight_snps_kwargs)

        starting_pos += max_pos + chrom_spacing

    plt.xticks(ticks, unique_chr)

    if x_col == 'POS':
        x_col = 'Genomic Position (BP)'

    plt.xlabel(x_col)

    if y_col == 'NLOG10_PVAL':
        y_col = "$-log_{10}$(p-value)"
    elif y_col == 'PVAL':
        y_col = "p-value"

    plt.ylabel(y_col)
    plt.tight_layout()


def qq_plot(input_data: Union[GWADataLoader, SumstatsTable],
            statistic='p_value'):
    """
    Generate a quantile-quantile (QQ) plot for the GWAS summary statistics.
    The function supports plotting QQ plots for the -log10(p-values) as well as
    the z-score (if available).

    :param input_data: An instance of `SumstatsTable` or `GWADataLoader` from which data about the
    positions of the SNPs will be extracted.
    :param statistic: The statistic to generate the QQ plot for. We currently support `p_value` and `z_score`.
    """

    import scipy.stats as stats

    if statistic == 'p_value':

        if isinstance(input_data, SumstatsTable):
            p_val = input_data.negative_log10_p_value
            m = input_data.m
        elif isinstance(input_data, GWADataLoader):
            p_val = np.concatenate([ss.negative_log10_p_value
                                    for ss in input_data.sumstats_table.values()])
            m = input_data.m
        else:
            raise ValueError("The input data must be an instance of `SumstatsTable` or `GWADataLoader`.")

        plt.scatter(-np.log10(np.arange(1, m + 1) / m), np.sort(p_val)[::-1])

        line = np.linspace(0., p_val.max() + 0.1*p_val.max())
        plt.plot(line, line, c='red')
        plt.xlabel("Expected $-log_{10}$(p-value)")
        plt.ylabel("Observed $-log_{10}$(p-value)")

    elif statistic == 'z_score':
        if isinstance(input_data, SumstatsTable):
            z_scs = input_data.z_score
        elif isinstance(input_data, GWADataLoader):
            z_scs = np.concatenate([ss.z_score for ss in input_data.sumstats_table.values()])
        else:
            raise ValueError("The input data must be an instance of `SumstatsTable` or `GWADataLoader`.")

        stats.probplot(z_scs, dist="norm", plot=plt)
        plt.show()
    else:
        raise NotImplementedError(f"No QQ plot can be generated for the statistic: {statistic}")

