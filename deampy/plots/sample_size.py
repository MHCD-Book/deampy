import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import deampy.plots.plot_support as Fig
import deampy.support.misc_functions as S
import deampy.support.misc_functions as mf
from deampy.plots.plot_support import output_figure


def plot_eff_sample_size(likelihood_weights, if_randomize=True,
                         fig_size=(6, 5), file_name=None,
                         title=None, x_label='Iteration', y_label='Effective Sample Size',
                         x_range=None, y_range=None
                         ):

    # convert the data to np array if needed
    if not type(likelihood_weights) == np.ndarray:
        likelihood_weights = np.array(likelihood_weights)

    # randomize the probabilities if needed
    if if_randomize:
        random.seed(1)
        random.shuffle(likelihood_weights)

    # calculate the effectiveve sample sizes through iterations
    effs = []
    for i in range(len(likelihood_weights)):
        effs.append(S.effective_sample_size(likelihood_weights[:i + 1]))

    # plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(range(1, len(likelihood_weights) + 1), effs)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # if f_star is not None:
    #     plt.axhline(y=f_star, linestyle='--', color='black', linewidth=1)
    Fig.output_figure(fig, file_name=file_name)


def plot_mean_stdev_by_np(
        list_of_obss, list_x_ranges, labels, colors,
        y_range_mean=None, y_range_stdev=None,
        y_label_mean=None, y_label_stdev=None,
        x_label=None, # conf_intv_transparency=0.2,
        true_mean=None, true_stdev=None, fig_size=None, file_name=None):
    """
    Plots the cumulative mean and standard deviation of a list of observations against sample size
    :param list_of_obss: list of observations (each observation is a list or np array)
    :param list_x_ranges: list of x ranges (each x range is a list or np array)
    :param labels: list of labels for each time series
    :param colors: list of colors for each time series
    :param y_range_mean: y-axis range for mean plot (list or tuple of form [ymin, ymax])
    :param y_range_stdev: y-axis range for standard deviation plot (list or tuple of form [ymin, ymax])
    :param y_label_mean: y-axis label for mean plot
    :param y_label_stdev: y-axis label for standard deviation plot
    :param x_label: x-axis label for both plots
    :param true_mean: true mean value to plot as a horizontal line (optional)
    :param true_stdev: true standard deviation value to plot as a horizontal line (optional)
    :param fig_size: figure size (tuple of form (width, height))
    :param file_name: file name to save the figure (optional)
    """

    f, axes = plt.subplots(1, 2, figsize=fig_size)
    axes[0].set_title('A)', loc='left', weight='bold')
    axes[1].set_title('B)', loc='left', weight='bold')

    for obs, color, label, x_range in zip(list_of_obss, colors, labels, list_x_ranges):

        cumulative_mean = mf.get_cumulative_mean(obs)
        cumulative_var = np.array(mf.get_cumulative_var(obs), dtype=float)

        axes[0].plot(x_range,
                   cumulative_mean, color=color, linewidth=1,
                   label=label)

        axes[1].plot(x_range,
                   np.sqrt(cumulative_var), color=color, linewidth=1,
                   label=label)
        #
        # if conf_intv_transparency is not None and conf_intv_transparency > 0:
        #     # plot 95% confidence interval for mean
        #     ci_lower, ci_upper = mf.get_confidence_interval_of_mean(obs, conf_level=0.95)
        #     axes[0].fill_between(x_range,
        #                          ci_lower,
        #                          ci_upper,
        #                          color=color,


    if true_mean is not None:
        axes[0].axhline(y=true_mean, color='k', linestyle='--', linewidth=1, label='True Value')

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label_mean)
    axes[0].set_ylim(y_range_mean)
    axes[0].legend(fontsize=8)
    # Format x-axis ticks with comma as thousands separator
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label_stdev)
    axes[1].set_ylim(y_range_stdev)

    if true_stdev is not None:
        axes[1].axhline(y=true_stdev,
                        color='k', linestyle='--', linewidth=1, label='True Variance')
    # Format x-axis ticks with comma as thousands separator
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))

    output_figure(plt=f, file_name=file_name)