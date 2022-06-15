from typing import Union
import magenpy as mgp
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats


def manhattan(gdl: Union[mgp.GWADataLoader, None] = None,
              sumstats: Union[mgp.SumstatsTable, None] = None,
              y=None,
              y_label=None,
              snp_color='#d0d0d0',
              snp_marker='o',
              snp_alpha=0.3,
              add_bonf_line=True,
              bonf_line_color='#b06a7a'):

    """
    Generate Manhattan plot where the x-axis is the genomic position (in BP)
    and the y-axis is the -log10(p-value) or some other statistic of the user's choice.

    TODO: Add functionality to highlight certain SNPs or markers on the plot.

    :param gdl: An instance of `GWADataLoader`.
    :param sumstats: An instance of `SumstatsTable`.
    :param y: A vector of values to plot on the y-axis.
    :param y_label: A label for the quantity or statistic that will be plotted.
    :param snp_color: The color of the dots on the Manhattan plot.
    :param snp_marker: The shape of the marker on the Manhattan plot.
    :param snp_alpha: The opacity level for the markers.
    :param add_bonf_line: If True, add a line indicating the Bonferroni significance threshold.
    :param bonf_line_color: The color of the Bonferroni significance threshold line.

    """

    if y is not None:
        assert y_label is not None

    if gdl is None:
        pos = {c: ss.bp_pos for c, ss in sumstats.split_by_chromosome()}
    else:
        pos = {c: ss.bp_pos for c, ss in gdl.sumstats_table.items()}

    starting_pos = 0
    ticks = []
    chrom_spacing = .1*min([p.max() - p.min() for c, p in pos.items()])

    if y is None:
        # If the values for the Y-axis are not provided,
        # we assume that the user wishes to plot a standard Manhattan plot
        # with -log10(p_value) on the Y-axis.

        if add_bonf_line:
            # Add bonferroni significance threshold line:
            plt.axhline(-np.log10(0.05 / 1e6), ls='--', zorder=1, color=bonf_line_color)

        if gdl is None:
            y = {c: ss.log10_p_value for c, ss in sumstats.split_by_chromosome()}
        else:
            y = {c: ss.log10_p_value for c, ss in gdl.sumstats_table.items()}

        y_label = "$-log_{10}$(p-value)"

    if gdl is None:
        unique_chr = sumstats.chromosomes
    else:
        unique_chr = gdl.chromosomes

    for i, c in enumerate(unique_chr):

        min_pos = pos[c].min()
        max_pos = pos[c].max()

        xmin = min_pos + starting_pos
        xmax = max_pos + starting_pos
        if i % 2 == 1:
            plt.axvspan(xmin=xmin, xmax=xmax, zorder=0, color='#808080')

        ticks.append((xmin + xmax) / 2)

        plt.scatter(pos[c] + starting_pos, y[c],
                    c=snp_color, alpha=snp_alpha, label=None,
                    marker=snp_marker)

        #if hl_snps is not None:
        #    plt.scatter((pos + starting_pos)[hl_snps[c]], y[c][hl_snps[c]],
        #                c=hl_snp_color, alpha=snp_alpha, label=hl_snp_label,
        #                marker=hl_snp_marker)

        starting_pos += max_pos + chrom_spacing

    plt.xticks(ticks, unique_chr)

    plt.xlabel("Genomic Position")
    plt.ylabel(y_label)

    plt.tight_layout()


def qq_plot(gdl: Union[mgp.GWADataLoader, None] = None,
            sumstats: Union[mgp.SumstatsTable, None] = None,
            statistic='p_value'):
    """
    Generate a quantile-quantile (QQ) plot for the GWAS summary statistics.
    The function supports plotting QQ plots for the -log10(p-values) as well as
    the z-score (if available).

    :param gdl: An instance of `GWADataLoader`.
    :param sumstats: An instance of `SumstatsTable`.
    :param statistic: The statistic to generate the QQ plot for. We currently support `p_value` and `z_score`.
    """

    if statistic == 'p_value':

        if gdl is None:
            p_val = sumstats.log10_p_value
            m = sumstats.m
        else:
            p_val = np.concatenate([ss.log10_p_value for ss in gdl.sumstats_table.values()])
            m = gdl.m

        plt.scatter(-np.log10(np.arange(1, m + 1) / m), np.sort(p_val)[::-1])

        line = np.linspace(0., p_val.max() + 0.1*p_val.max())
        plt.plot(line, line, c='red')
        plt.xlabel("Expected $-log_{10}$(p-value)")
        plt.ylabel("Observed $-log_{10}$(p-value)")

    elif statistic == 'z_score':
        if gdl is None:
            z_scs = sumstats.z_score
        else:
            z_scs = np.concatenate([ss.z_score for ss in gdl.sumstats_table.values()])

        stats.probplot(z_scs, dist="norm", plot=plt)
        plt.show()
    else:
        raise ValueError(f"No QQ plot can be generated for the statistic: {statistic}")


def plot_ld_matrix(ldm: mgp.LDMatrix, row_slice=None, col_slice=None):
    """
    Plot a heatmap representing the LD matrix or portions of it.

    :param ldm: An instance of `LDMatrix`.
    :param row_slice: A `slice` object indicating which rows to extract from the LD matrix.
    :param col_slice: A `slice` object indicating which columns to extract from the LD matrix.
    """

    plt.imshow(ldm.to_csr_matrix()[row_slice, col_slice].toarray())
    plt.colorbar()
