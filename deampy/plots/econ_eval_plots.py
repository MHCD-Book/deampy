import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from deampy.plots.plot_support import output_figure, calculate_ticks, format_axis_tick_labels


def format_ax(ax,
              x_range=None, x_delta=None,
              y_range=None, y_delta=None, if_y_axis_prob=False,
              x_axis_format_deci=None, y_axis_format_deci=None):

    # the range of x and y-axis are set so that we can get the
    # tick values and label
    if y_range is None and if_y_axis_prob:
        ax.set_ylim((-0.01, 1.01))
    if y_range:
        ax.set_ylim(y_range)
    if x_range:
        ax.set_xlim(x_range)

    # get x ticks
    if x_delta is None:
        vals_x = ax.get_xticks()
    else:
        vals_x = calculate_ticks(interval=x_range, delta=x_delta)

    # get y ticks
    if y_delta is None:
        vals_y = ax.get_yticks()
    else:
        vals_y = calculate_ticks(interval=y_range, delta=y_delta)

    # format x-axis
    ax.set_xticks(vals_x)
    if x_axis_format_deci is None:
        format_axis_tick_labels(ax=ax, axis='x', format_deci=(',', 0))
    else:
        format_axis_tick_labels(ax=ax, axis='x', format_deci=x_axis_format_deci)

    # d = 2 * (x_range[1] - x_range[0]) / 200
    # ax.set_xlim([x_range[0] - d, x_range[1] + d])

    # format y-axis
    # if y_range is None:
    ax.set_yticks(vals_y)
    if if_y_axis_prob:
        format_axis_tick_labels(ax=ax, axis='y', format_deci=(None, 1))
    if y_axis_format_deci is not None:
        format_axis_tick_labels(ax=ax, axis='y', format_deci=y_axis_format_deci)

    if y_range is None and if_y_axis_prob:
        ax.set_ylim((-0.01, 1.01))
    elif y_range:
        ax.set_ylim(y_range)
    else:
        ax.set_ylim([vals_y[0], vals_y[-1]])

    if not if_y_axis_prob:
        ax.axhline(y=0, c='k', ls='--', linewidth=0.5)


def add_grids(ax, grid_info):

    # grid
    if grid_info is None:
        pass
    elif grid_info == 'default':
        color, linestyle, linewidth, alpha = ('k', '--', 0.5, 0.2)
    else:
        color, linestyle, linewidth, alpha = grid_info
    if grid_info is not None:
        ax.grid(color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)


def add_curves_to_ax(ax, curves, title=None,
                     x_range=None, x_label=None, y_label=None, y_range=None,
                     y_axis_multiplier=1, y_axis_decimal=1,
                     x_delta=None,
                     transparency_lines=0.5,
                     transparency_intervals=0.2,
                     legends=None,
                     show_legend=False,
                     show_frontier=True,
                     show_labels_on_frontier=False,
                     curve_line_width=1.0, frontier_line_width=4.0,
                     if_y_axis_prob=False,
                     if_format_y_numbers=True,
                     legend_font_size_and_loc=(7, 'upper left'),
                     frontier_label_shift_x=-0.01,
                     frontier_label_shift_y=0.01,
                     grid_info=None):

    for i, curve in enumerate(curves):

        label = curve.label if legends is None else legends

        # if to interpolate points to create a smooth curve
        interpolate = False
        if interpolate:
            f2 = interp1d(curve.xs, curve.ys * y_axis_multiplier, kind='cubic')
            x_smooth = np.linspace(curve.xs.min(), curve.xs.max(), 100)
            y_smooth = f2(x_smooth)
            xs = x_smooth
            ys = y_smooth
        else:
            xs = curve.xs
            ys = curve.ys * y_axis_multiplier

        # plot line
        ax.plot(xs, ys,
                c=curve.color, alpha=transparency_lines,
                linewidth=curve_line_width, linestyle=curve.linestyle, label=label)

        # plot intervals
        if curve.l_errs is not None and curve.u_errs is not None:
            ax.fill_between(curve.xs,
                            (curve.ys - curve.l_errs) * y_axis_multiplier,
                            (curve.ys + curve.u_errs) * y_axis_multiplier,
                            color=curve.color, alpha=transparency_intervals)
        # plot frontier
        if show_frontier:
            # check if this strategy is not dominated
            if curve.frontierXs is not None and len(curve.frontierXs) > 0:
                y = [y*y_axis_multiplier if y is not None else None for y in curve.frontierYs]
                ax.plot(curve.frontierXs, y,
                        c=curve.color, alpha=1, linewidth=frontier_line_width)

    if show_legend:
        ax.legend(fontsize=legend_font_size_and_loc[0], loc=legend_font_size_and_loc[1]) #loc=2,

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_range)

    # add labels on the frontier
    if show_labels_on_frontier:
        y_min, y_max = ax.get_ylim()
        y_axis_length = y_max - y_min
        for curve in curves:
            if curve.frontierXs is not None and len(curve.frontierXs) > 0:
                if curve.frontierYs[0] is not None and curve.frontierYs[-1] is not None:
                    x_axis_length = x_range[1] - x_range[0]
                    x = 0.5 * (curve.frontierXs[0] + curve.frontierXs[-1]) + frontier_label_shift_x * x_axis_length
                    y = 0.5 * (curve.frontierYs[0] + curve.frontierYs[-1]) * y_axis_multiplier \
                        + frontier_label_shift_y * y_axis_length
                    ax.text(x=x, y=y, s=curve.label, fontsize=legend_font_size_and_loc + 1, c=curve.color)

    # grids
    add_grids(ax=ax, grid_info=grid_info)

    # do the other formatting
    format_ax(ax=ax, y_range=y_range,
              x_range=x_range, x_delta=x_delta,
              if_y_axis_prob=if_y_axis_prob,
              y_axis_format_deci=[',', y_axis_decimal])


def add_min_monte_carlo_samples_to_ax(
        ax, dict_of_ns, epsilons, x_range=None, y_range=None, x_multiplier=1):

    colors = ('purple', 'blue', 'green', 'red')
    markers = ('o', 'v', '^', 's')

    adjusted_epsilons = [x*x_multiplier for x in epsilons]

    i = 0
    for alpha in dict_of_ns:
        ns = [dict_of_ns[alpha][key] for key in dict_of_ns[alpha]]

        # build n values and intervals
        n_values = []
        l_errs = []
        u_errs = []
        for n in ns:
            if isinstance(n, tuple):
                value, interval = n
                n_values.append(value)
                l_errs.append(value - interval[0])
                u_errs.append(interval[1] - value)
            else:
                n_values.append(n)

        ax.scatter(adjusted_epsilons, n_values, marker=markers[i], color=colors[i],
                   label=r'$\alpha=${:.{prec}%}'.format(alpha, prec=0))
        # error bars
        if len(l_errs) > 0:
            ax.errorbar(
                adjusted_epsilons, n_values, yerr=[l_errs, u_errs], fmt='none',
                ecolor=colors[i], capsize=3, linewidth=0.75)

        ax.plot(adjusted_epsilons, n_values, 'k--', color=colors[i], linewidth=0.5)
        i += 1

    if y_range:
        ax.set_ylim(y_range)
    else:
        ax.set_ylim(0)

    if x_range:
        ax.set_xlim(x_range)

    ax.set_xticks(adjusted_epsilons)
    vals_x = ax.get_xticks()
    ax.set_xticklabels(['${:,.{prec}f}'.format(x, prec=0) for x in vals_x])

    vals_y = ax.get_yticks()
    ax.set_yticklabels(['{:,.{prec}f}'.format(y, prec=0) for y in vals_y])

    ax.legend(fontsize=8)


def add_icer_over_itr_to_ax(ax, icer_over_itr, x_range=None, y_range=None, y_delta=None):

    ax.plot(icer_over_itr, 'b-', linewidth=0.5)

    # if y_range:
    #     ax.set_ylim(y_range)
    #
    # if x_range:
    #     ax.set_xlim(x_range)

    format_ax(ax=ax,
              x_range=x_range, x_axis_format_deci=(',', 0),
              y_range=y_range, y_delta=y_delta, y_axis_format_deci=('$', 0))
    #
    #
    # format_axis_tick_labels(ax=ax, axis='x', format_deci=(',', 0))
    #
    #
    # if y_delta is None:
    #     vals_y = ax.get_yticks()
    # else:
    #     vals_y = calculate_ticks(interval=y_range, delta=y_delta)
    # ax.set_yticks(vals_y)
    # ax.set_yticklabels(['${:,.0f}'.format(y) for y in vals_y])


def plot_icer_over_itrs(icer_over_itr, x_range=None, y_range=None, legend=None, figure_size=None, file_name=None):

    fig, ax = plt.subplots(figsize=figure_size)

    # add icer over iterations
    add_icer_over_itr_to_ax(ax=ax, icer_over_itr=icer_over_itr, x_range=x_range, y_range=y_range)

    # add legend if provided
    if legend is not None:
        ax.legend([legend])

    # output figure
    output_figure(fig, file_name)
