import inspect

import matplotlib.pyplot as plt
import numpy as np
from numpy import iinfo, int32

from deampy.plots.histogram import add_histogram_to_ax
from deampy.plots.plot_support import output_figure, get_moving_average


class MCMC:

    def __init__(self, prior_ranges, std_factor=0.1):
        self.priorRanges = prior_ranges
        self.stdFactors = [(r[1]-r[0])*std_factor for r in prior_ranges]
        self.samples = [[] for i in range(len(prior_ranges))]  # Initialize samples for each parameter
        self.seeds = []

    def run(self, log_likelihood_func, num_samples=1000, rng=None):
        """Run a simple Metropolis-Hastings MCMC algorithm."""

        # assert that log_likelihood_func is callable
        if not callable(log_likelihood_func):
            raise ValueError("log_likelihood_func must be a callable function.")
        # assert that log_likelihood_func accepts two parameters: thetas and seed
        sig = inspect.signature(log_likelihood_func)
        if len(sig.parameters) != 2 or 'thetas' not in sig.parameters or 'seed' not in sig.parameters:
            raise ValueError("log_likelihood_func must accept two parameters: thetas and seed.")

        if rng is None:
            rng = np.random.RandomState(1)

        seed = rng.randint(0, iinfo(int32).max)

        # Start from a uniform prior
        thetas = np.array(
            [rng.uniform(low=prior_range[0], high=prior_range[1]) for prior_range in self.priorRanges])

        log_prior_value = self._log_prior(thetas=thetas)

        log_post = log_prior_value + log_likelihood_func(thetas=thetas, seed=seed)

        for _ in range(num_samples):

            seed = rng.randint(0, iinfo(int32).max)

            thetas_prop = rng.normal(thetas, self.stdFactors)
            log_post_prop = (
                    self._log_prior(thetas=thetas_prop)
                    + log_likelihood_func(thetas=thetas_prop, seed=seed))

            accept_prob = min(1, np.exp(log_post_prop - log_post))

            if rng.random() < accept_prob:
                thetas = thetas_prop
                log_post = log_post_prop

            self.seeds.append(seed)
            for i in range(len(self.priorRanges)):
                self.samples[i].append(thetas[i])

    def _log_prior(self, thetas):
        """Compute the log-prior of theta."""

        log_prior = 0
        for i, theta in enumerate(thetas):
            if self.priorRanges[i][0] <= theta <= self.priorRanges[i][1]:
                log_prior += np.log(1 / (self.priorRanges[i][1] - self.priorRanges[i][0]))
            else:
                return -np.inf  # Outside prior range, log-prior is -inf
        return log_prior

    @staticmethod
    def add_trace_to_ax(ax, samples, par_name, moving_ave_window, y_range=None):

        ax.plot(samples, label=par_name)
        if moving_ave_window is not None:
            ax.plot(get_moving_average(samples, window=moving_ave_window),
                    label=f'Moving Average ({moving_ave_window})', color='k', linestyle='--')
        ax.set_title(par_name)
        ax.set_ylim(y_range)

    def plot_trace(self, n_rows=1, n_cols=1, figsize=(7, 5),
                   file_name=None, share_x=False, share_y=False,
                   parameter_names=None, moving_ave_window=None):
        """Plot the trace of the MCMC samples."""

        # plot each panel
        f, axarr = plt.subplots(n_rows, n_cols, sharex=share_x, sharey=share_y, figsize=figsize)

        if parameter_names is None:
            parameter_names = [f'Parameter {i+1}' for i in range(len(self.priorRanges))]

        for i in range(n_rows):
            for j in range(n_cols):
                # get current axis
                if n_rows == 1 or n_cols == 1:
                    ax = axarr[i * n_cols + j]
                else:
                    ax = axarr[i, j]

                # plot subplot, or hide extra subplots
                if i * n_cols + j >= len(self.samples):
                    ax.axis('off')
                else:
                    self.add_trace_to_ax(
                        ax=ax,
                        samples=self.samples[i * n_cols + j],
                        par_name=parameter_names[i * n_cols + j],
                        moving_ave_window=moving_ave_window,
                        y_range=self.priorRanges[i * n_cols + j]
                    )

                    ax.set_xlabel('Step')
                    ax.set_ylabel('Sample Value')

                # remove unnecessary labels for shared axis
                if share_x and i < n_rows - 1:
                    ax.set(xlabel='')
                if share_y and j > 0:
                    ax.set(ylabel='')

        output_figure(plt=f, file_name=file_name)

    def plot_posterior(self, n_warmup, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):
        """Plot the posterior distribution of the MCMC samples."""

        # plot each panel
        f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

        if parameter_names is None:
            parameter_names = [f'Parameter {i+1}' for i in range(len(self.priorRanges))]

        for i in range(n_rows):
            for j in range(n_cols):
                # get current axis
                if n_rows == 1 or n_cols == 1:
                    ax = axarr[i * n_cols + j]
                else:
                    ax = axarr[i, j]

                # plot subplot, or hide extra subplots
                if i * n_cols + j >= len(self.samples):
                    ax.axis('off')
                else:
                    add_histogram_to_ax(
                        ax=ax,
                        data=self.samples[i * n_cols + j][n_warmup:],  # Skip warmup samples
                        # color='blue',
                        title=parameter_names[i * n_cols + j],
                        x_label='Sampled Values',
                        # y_label=None,
                        x_range=self.priorRanges[i * n_cols + j],
                        y_range=None,
                        transparency=0.7,
                    )

        output_figure(plt=f, file_name=file_name)