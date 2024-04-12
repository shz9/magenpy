from typing import Union
from ..GWADataLoader import GWADataLoader
from ..SumstatsTable import SumstatsTable
import matplotlib.pylab as plt
import numpy as np


def manhattan(input_data: Union[GWADataLoader, SumstatsTable],
              y=None,
              y_label=None,
              scatter_kwargs=None,
              #highlight_snps=None,
              #highlight_snps_kwargs=None,
              chrom_sep_color='#f0f0f0',
              add_bonf_line=True,
              bonf_line_kwargs=None):

    """
    Generate Manhattan plot where the x-axis is the genomic position (in BP)
    and the y-axis is the -log10(p-value) or some other statistic of the user's choice.

    TODO: Add functionality to highlight certain SNPs or markers on the plot.
    TODO: Allow the user to plot other statistics on the y-axis.

    :param input_data: An instance of `SumstatsTable` or `GWADataLoader` from which data about the
    positions of the SNPs will be extracted.
    :param y: An optional vector of values to plot on the y-axis. If not provided, the -log10(p-value)
    will be plotted by default.
    :param y_label: A label for the quantity or statistic that will be plotted on the y-axis.
    :param chrom_sep_color: The color for the chromosome separator block.
    :param scatter_kwargs: A dictionary of keyword arguments to pass to the `plt.scatter` function.
    This can be used to customize the appearance of the points on the scatter plot.
    :param add_bonf_line: If True, add a line indicating the Bonferroni significance threshold.
    :param bonf_line_kwargs: The color of the Bonferroni significance threshold line.

    """

    if isinstance(input_data, SumstatsTable):
        pos = {c: ss.bp_pos for c, ss in input_data.split_by_chromosome().items()}
    elif isinstance(input_data, GWADataLoader):
        pos = {c: ss.bp_pos for c, ss in input_data.sumstats_table.items()}
    else:
        raise ValueError("The input data must be an instance of `SumstatsTable` or `GWADataLoader`.")

    # -------------------------------------------------------
    # Add custom scatter plot arguments (if not provided)
    if scatter_kwargs is None:
        scatter_kwargs = {'marker': '.', 'alpha': 0.3, 'color': '#808080'}
    else:
        # Only update the keys that are not already present in the dictionary:
        scatter_kwargs = {**scatter_kwargs, **{'marker': '.', 'alpha': 0.3, 'color': '#808080'}}

    # Add custom Bonferroni line arguments (if not provided)
    if bonf_line_kwargs is None:
        bonf_line_kwargs = {'color': '#b06a7a', 'ls': '--', 'zorder': 1}
    else:
        # Only update the keys that are not already present in the dictionary:
        bonf_line_kwargs = {**bonf_line_kwargs, **{'color': '#b06a7a', 'ls': '--', 'zorder': 1}}

    # -------------------------------------------------------

    starting_pos = 0
    ticks = []
    chrom_spacing = .1*min([p.max() - p.min() for c, p in pos.items()])

    if y is None:
        # If the values for the Y-axis are not provided,
        # we assume that the user wishes to plot a standard Manhattan plot
        # with -log10(p_value) on the Y-axis.

        if add_bonf_line:
            # Add Bonferroni significance threshold line:
            plt.axhline(-np.log10(0.05 / 1e6), bonf_line_kwargs)

        if isinstance(input_data, SumstatsTable):
            y = {c: ss.log10_p_value for c, ss in input_data.split_by_chromosome().items()}
        else:
            y = {c: ss.log10_p_value for c, ss in input_data.sumstats_table.items()}

        y_label = "$-log_{10}$(p-value)"

    unique_chr = sorted(list(pos.keys()))

    for i, c in enumerate(unique_chr):

        min_pos = pos[c].min()
        max_pos = pos[c].max()

        xmin = min_pos + starting_pos
        xmax = max_pos + starting_pos
        if i % 2 == 1:
            plt.axvspan(xmin=xmin, xmax=xmax, zorder=0, color=chrom_sep_color)

        ticks.append((xmin + xmax) / 2)

        plt.scatter(pos[c] + starting_pos,
                    y[c],
                    scatter_kwargs)

        #if hl_snps is not None:
        #    plt.scatter((pos + starting_pos)[hl_snps[c]], y[c][hl_snps[c]],
        #                c=hl_snp_color, alpha=snp_alpha, label=hl_snp_label,
        #                marker=hl_snp_marker)

        starting_pos += max_pos + chrom_spacing

    plt.xticks(ticks, unique_chr)

    plt.xlabel("Genomic Position")
    plt.ylabel(y_label)

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
            p_val = input_data.log10_p_value
            m = input_data.m
        elif isinstance(input_data, GWADataLoader):
            p_val = np.concatenate([ss.log10_p_value for ss in input_data.sumstats_table.values()])
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

