import os
import string

from deampy.support.misc_functions import *


def output_figure(plt, file_name=None, dpi=300, tight_layout=True, bbox_inches='tight'):
    """
    :param plt: reference to the plot
    :param file_name: filename to save this figure as (e.g. 'figure.png') (if None, the figure will be displayed)
    :param dpi: dpi of the figure
    """

    if tight_layout:
        plt.tight_layout()

    # output
    if file_name is None or file_name == '':
        plt.show()
    else:
        if file_name[0] == '/':
            file_name = file_name[1:]
        # get directory
        directory_path = os.path.dirname(file_name)

        # create the directory if does not exist
        if directory_path != '':
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

        try:
            plt.savefig(proper_file_name(file_name), dpi=dpi, bbox_inches=bbox_inches)
        except Exception as e:
            raise ValueError("Error in saving figure '{}'. "
                             "Ensure that the filename is valid. Error Message: {}".format(file_name, str(e)))

        # try:
        #     plt.savefig(proper_file_name(filename), dpi=dpi, bbox_inches=bbox_inches)
        # except ValueError:
        #     raise ValueError("Error in saving figure '{}'. "
        #                      "Ensure that the filename is valid and "
        #                      "that the folder where the figure should be saved exists.".format(filename))


def calculate_ticks(interval, delta):
    values = []
    v = interval[0]
    while v <= interval[1]:
        values.append(v)
        v += delta

    return values


def format_axis_tick_labels(ax, axis='x', format_deci=None):

    if axis == 'x':
        vals = ax.get_xticks()
        if format_deci[0] is None or format_deci[0] == '':
            ax.set_xticklabels(['{:.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == ',':
            ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '$':
            ax.set_xticklabels(['${:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '%':
            ax.set_xticklabels(['{:,.{prec}%}'.format(x, prec=format_deci[1]) for x in vals])

    elif axis == 'y':
        vals = ax.get_yticks()
        if format_deci[0] is None or format_deci[0] == '':
            ax.set_yticklabels(['{:.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == ',':
            ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '$':
            ax.set_yticklabels(['${:,.{prec}f}'.format(x, prec=format_deci[1]) for x in vals])
        elif format_deci[0] == '%':
            ax.set_yticklabels(['{:,.{prec}%}'.format(x, prec=format_deci[1]) for x in vals])
    else:
        raise ValueError("axis must be either 'x' or 'y'.")


# def format_x_axis(ax, min_x, max_x, delta_x, buffer=0, form=None, deci=None):
#
#     # get x ticks
#     xs = ax.get_xticks()
#
#     if min_x is None:
#         min_x = xs[0]
#     if max_x is None:
#         max_x = xs[-1]
#     if delta_x is not None:
#         xs = calculate_ticks(min_x, max_x, delta_x)
#
#     # format x-axis
#     ax.set_xticks(xs)
#     if deci is not None:
#         ax.set_xticklabels([F.format_number(x, deci=deci, format=form) for x in xs])
#
#     ax.set_xlim([min_x-buffer, max_x+buffer])


def add_labels_to_panels(axarr, x_coord=-0.2, y_coord=1.1, font_size=8):
    """
    adds A), B), etc. labels to panels
    :param axarr: (array) of panels
    :param x_coord: (float) increase to move labels right
    :param y_coord: (float) increase to move labels up
    :param font_size: (float) font size of labels
    """
    axs = axarr.flat
    for n, ax in enumerate(axs):
        ax.text(x_coord, y_coord,
                string.ascii_uppercase[n] + ')',
                transform=ax.transAxes,
                size=font_size,
                weight='bold')
