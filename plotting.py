import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats


def plot_manhattan(gdl,
                   y=None,
                   y_label=None,
                   title=None,
                   output_fname=None,
                   snp_color='#d0d0d0',
                   snp_marker='o',
                   hl_snps=None,
                   hl_snp_color='#cc3300',
                   hl_snp_marker='*',
                   hl_snp_label=None,
                   snp_alpha=0.3,
                   add_bonf_line=True,
                   bonf_line_color='#b06a7a'):

    starting_pos = 0
    ticks = []
    chrom_spacing = 25000000

    plt.figure(figsize=(12, 6))

    if y is None:
        if add_bonf_line:
            # Add bonferroni significance threshold line:
            plt.axhline(-np.log10(0.05 / gdl.M), ls='--', zorder=1,
                        color='#263640')

        y = {c: -np.log10(pval) for c, pval in gdl.p_values.items()}
        y_label = "$-log_{10}(p-value)$"

    unique_chr = gdl.genotype_index

    for i, c in enumerate(unique_chr):

        pos = gdl.genotypes[c].pos.values
        max_pos = pos.max()

        xmin = starting_pos #- (chrom_spacing / 2)
        xmax = max_pos + starting_pos #+ (chrom_spacing / 2)
        if i % 2 == 1:
            plt.axvspan(xmin=xmin, xmax=xmax, zorder=0, color='#808080')

        # TODO: Fix tick positioning
        ticks.append((xmin + xmax) / 2)

        plt.scatter(pos + starting_pos, y[c],
                    c=snp_color, alpha=snp_alpha, label=None,
                    marker=snp_marker)

        if hl_snps is not None:
            plt.scatter((pos + starting_pos)[hl_snps[c]], y[c][hl_snps[c]],
                        c=hl_snp_color, alpha=snp_alpha, label=hl_snp_label,
                        marker=hl_snp_marker)

        starting_pos += max_pos #+ chrom_spacing

    plt.xticks(ticks, unique_chr)

    plt.xlabel("Genomic Position")
    plt.ylabel(y_label)

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if output_fname is not None:
        plt.savefig(output_fname)
    else:
        plt.show()

    plt.close()


def plot_qq(gdl, quantity='p_value'):

    if quantity == 'p_value':
        p_vals = -np.log10(np.sort(np.concatenate(list(gdl.p_values.values()))))
        plt.scatter(-np.log10(np.arange(1, gdl.M + 1) / gdl.M),
                    p_vals)

        line = np.linspace(0., p_vals.max() + 0.1*p_vals.max())
        plt.plot(line, line, c='red')
        plt.xlabel("Expected $-log_{10}(P-value)$")
        plt.ylabel("Observed $-log_{10}(P-value)$")

    elif quantity == 'z_score':
        z_scs = np.concatenate(list(gdl.z_scores.values()))
        stats.probplot(z_scs, dist="norm", plot=plt)
        plt.show()

    else:
        raise ValueError(f"No QQ plot can be generated for this quantity: {quantity}")


def plot_ld_matrix(ld, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(ld)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()
