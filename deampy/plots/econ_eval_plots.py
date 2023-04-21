import numpy as np
from scipy.interpolate import interp1d

def format_ax(ax,
              x_range=None, x_delta=None,
              y_range=None, y_delta=None, if_y_axis_prob=True,
              if_format_y_numbers=True, y_axis_decimal=1):

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
        vals_x = []
        x = x_range[0]
        while x <= x_range[1]:
            vals_x.append(x)
            x += x_delta

    # get y ticks
    if y_delta is None:
        vals_y = ax.get_yticks()
    else:
        vals_y = []
        y = y_range[0]
        while y <= y_range[1]:
            vals_y.append(y)
            y += y_delta

    # format x-axis
    ax.set_xticks(vals_x)
    ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

    d = 2 * (x_range[1] - x_range[0]) / 200
    ax.set_xlim([x_range[0] - d, x_range[1] + d])

    # format y-axis
    if y_range is None:
        ax.set_yticks(vals_y)
    if if_y_axis_prob:
        ax.set_yticklabels(['{:.{prec}f}'.format(x, prec=1) for x in vals_y])
    elif if_format_y_numbers:
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=y_axis_decimal) for x in vals_y])

    if y_range is None and if_y_axis_prob:
        ax.set_ylim((-0.01, 1.01))
    if y_range:
        ax.set_ylim(y_range)

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
                     legend_font_size=7,
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
        ax.legend(fontsize=legend_font_size) #loc=2,

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
                    ax.text(x=x, y=y, s=curve.label, fontsize=legend_font_size+1, c=curve.color)

    # grids
    add_grids(ax=ax, grid_info=grid_info)

    # do the other formatting
    format_ax(ax=ax, y_range=y_range,
              x_range=x_range, x_delta=x_delta,
              if_y_axis_prob=if_y_axis_prob,
              if_format_y_numbers=if_format_y_numbers,
              y_axis_decimal=y_axis_decimal)


def add_min_monte_carlo_samples_to_ax(
        ax, dict_of_ns, epsilons, x_range=None, y_range=None, x_multiplier=1):

    colors = ('purple', 'blue', 'green', 'red')
    markers = ('o', 'v', '^', 's')

    i = 0
    for alpha in dict_of_ns:
        ns = [dict_of_ns[alpha][key] for key in dict_of_ns[alpha]]
        ax.scatter(epsilons, ns, marker=markers[i], color=colors[i],
                   label=r'$\alpha=${:.{prec}%}'.format(alpha, prec=0))
        ax.plot(epsilons, ns, 'k--', color=colors[i], linewidth=0.5)
        i += 1

    if y_range:
        ax.set_ylim(y_range)
    else:
        ax.set_ylim(0)

    ax.set_xticks(epsilons)
    vals_x = ax.get_xticks()
    ax.set_xticklabels(['${:,.{prec}f}'.format(x*x_multiplier, prec=0) for x in vals_x])

    vals_y = ax.get_yticks()
    ax.set_yticklabels(['{:,.{prec}f}'.format(y, prec=0) for y in vals_y])

    if x_range:
        ax.set_xlim(x_range)

    ax.legend(fontsize=8)
