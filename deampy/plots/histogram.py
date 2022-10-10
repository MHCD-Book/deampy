import matplotlib.pyplot as plt
import numpy as np

from deampy.plots.plot_support import output_figure


def add_histogram_to_ax(ax, data, title=None, color=None, bin_width=None, x_range=None,
                        transparency=1.0, label=None, format_deci=None, remove_y_labels=False,
                        linewidth=1, x_delta=None):
    """
    :param ax: (axis)
    :param data: (list) of observations
    :param title: (strig) title of the histogram
    :param color:
    :param bin_width:
    :param x_range:
    :param transparency:
    :param label:
    :param format_deci: [a, b] where a could be ',', '$', or '%' and b is the decimal point
    :param remove_y_labels: (bool) set to True to remove ticks and labels of the y-axis
    :param linewidth: (double)
    :return:
    """

    ax.hist(data,
            bins=find_bins(data, x_range, bin_width),
            color=color,
            edgecolor='black',
            linewidth=linewidth,
            alpha=transparency,
            label=label)
    ax.set_xlim(x_range)
    ax.set_title(title)
    ax.yaxis.set_visible(not remove_y_labels)

    if x_delta:
        vals_x = []
        if x_delta:
            vals_x = []
            x = x_range[0]
            while x <= x_range[1]:
                vals_x.append(x)
                x += x_delta
        ax.set_xticks(vals_x)

    if format_deci is not None:
        vals = ax.get_xticks()
        if format_deci[0] is None or format_deci[0] == '':
            ax.set_xticklabels(['{:.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == ',':
            ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '$':
            ax.set_xticklabels(['${:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '%':
            ax.set_xticklabels(['{:,.{prec}%}'.format(x, prec=format_deci[1]) for x in vals])


def plot_histogram(data, title=None,
                   x_label=None, y_label=None, bin_width=None,
                   x_range=None, y_range=None, figure_size=None,
                   color=None, legend=None, file_name=None):
    """ plot a histogram
    :param data: (list) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param figure_size: (tuple) figure size
    :param color: (string) color
    :param legend: string for the legend
    :param file_name: (string) filename to save the histogram as (e.g. 'fig.png')
    """

    fig, ax = plt.subplots(figsize=figure_size)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add histogram
    add_histogram_to_ax(ax=ax,
                        data=data,
                        color=color,
                        bin_width=bin_width,
                        x_range=x_range,
                        transparency=0.75)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # add legend if provided
    if legend is not None:
        ax.legend([legend])

    # output figure
    output_figure(fig, file_name)


def plot_histograms(data_sets, legends, bin_width=None,
                    title=None, x_label=None, y_label=None,
                    x_range=None, y_range=None, figure_size=None,
                    color_codes=None, transparency=1, file_name=None):
    """
    plots multiple histograms on a single figure
    :param data_sets: (list of lists) observations
    :param legends: (list) string for the legend
    :param bin_width: bin width
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param figure_size: (tuple) figure size
    :param color_codes: (list) of colors
    :param transparency: (float) 0.0 transparent through 1.0 opaque
    :param file_name: (string) filename to to save the histogram as (e.g. 'fig.png')
    """

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add histograms
    for i, data in enumerate(data_sets):
        color = None
        if color_codes is not None:
            color = color_codes[i]

        add_histogram_to_ax(ax=ax,
                            data=data,
                            bin_width=bin_width,
                            x_range=x_range,
                            color=color,
                            transparency=transparency,
                            label=legends[i])

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend()

    # output figure
    output_figure(plt, file_name)


def find_bins(data, x_range, bin_width):

    if bin_width is None:
        return 'auto'

    if x_range is not None:
        l = x_range[0]
        u = x_range[1]
    else:
        l = min(data)
        u = max(data) + bin_width
    return np.arange(l, u, bin_width)