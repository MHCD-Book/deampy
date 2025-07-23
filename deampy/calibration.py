import inspect

import matplotlib.pyplot as plt
import numpy as np

import deampy.in_out_functions as IO
from deampy.format_functions import format_number, format_interval
from deampy.plots.histogram import add_histogram_to_ax
from deampy.plots.plot_support import output_figure, get_moving_average
from deampy.statistics import SummaryStat


class _Calibration:

    def __init__(self, prior_ranges):
        """Base class for calibration methods."""

        assert isinstance(prior_ranges, dict) and len(prior_ranges) > 0, \
            "prior_ranges must be a non-empty dictionary of tuples (min, max) for each parameter."

        self.samples = None
        self.seeds = None
        self.logLikelihoods = None

        self.priorRanges = prior_ranges  # List of tuples (min, max) for each parameter
        self._reset()

    def _reset(self):
        """Reset the calibration object."""
        self.samples = [[] for i in range(len(self.priorRanges))] # Initialize samples for each parameter
        self.seeds = []
        self.logLikelihoods = []

    def run(self, *args, **kwargs):
        """Run the calibration method."""
        raise NotImplementedError("Subclasses should implement this method.")

    def save_samples(self, file_name, parameter_names=None):

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        # first row
        first_row = ['Seed', 'Log-Likelihood']
        first_row.extend(parameter_names)

        # produce the list to report the results
        csv_rows = [first_row]

        for i in range(len(self.seeds)):
            # create a row with seed, log-likelihood, and parameter samples
            row = [self.seeds[i], self.logLikelihoods[i]]
            row.extend([self.samples[j][i] for j in range(len(self.priorRanges))])

            csv_rows.append(row)

        # write the calibration result into a csv file
        IO.write_csv(
            file_name=file_name,
            rows=csv_rows)

    def read_samples(self, file_name):
        """Read samples from a CSV file."""

        self._reset()

        cols = IO.read_csv_cols(file_name=file_name, if_ignore_first_row=True, if_convert_float=True)

        # the first column is seeds
        self.seeds = cols[0].astype(int).tolist()
        # the second column is log-likelihoods
        self.logLikelihoods = cols[1].tolist()
        # remaining columns are parameter samples
        for i in range(len(self.priorRanges)):
            self.samples[i].extend(cols[i + 2].tolist())

    @staticmethod
    def _get_probs(likelihoods):
        """Normalize the weights to sum to 1."""

        l_max = np.max(likelihoods)
        likelihoods = likelihoods - l_max
        weights = np.exp(likelihoods)
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("All likelihoods are zero, cannot normalize probabilities.")
        normalized_probs = weights / weights_sum
        return normalized_probs

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
            parameter_names = list(self.priorRanges.keys())

        i = 0
        j = 0
        for key, range in self.priorRanges.items():

            # get current axis
            if n_rows == 1 and n_cols == 1:
                ax = axarr
            elif n_rows == 1 or n_cols == 1:
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
                    y_range=range
                )

                ax.set_xlabel('Step')
                ax.set_ylabel('Sample Value')

            # remove unnecessary labels for shared axis
            if share_x and i < n_rows - 1:
                ax.set(xlabel='')
            if share_y and j > 0:
                ax.set(ylabel='')

            if j == n_cols - 1:
                i += 1
                j = 0
            else:
                j += 1

        output_figure(plt=f, file_name=file_name)

    def _plot_posterior(self, samples, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):
        """Plot the posterior distribution of the MCMC samples."""

        # plot each panel
        f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        i = 0
        j = 0
        for key, range in self.priorRanges.items():

            # get current axis
            if n_rows == 1 and n_cols == 1:
                ax = axarr
            elif n_rows == 1 or n_cols == 1:
                ax = axarr[i * n_cols + j]
            else:
                ax = axarr[i, j]

            # plot subplot, or hide extra subplots
            if i * n_cols + j >= len(samples):
                ax.axis('off')
            else:
                add_histogram_to_ax(
                    ax=ax,
                    data=samples[i * n_cols + j],  # Skip warmup samples
                    # color='blue',
                    title=parameter_names[i * n_cols + j],
                    x_label='Sampled Values',
                    # y_label=None,
                    x_range=range,
                    y_range=None,
                    transparency=0.7,
                )

            if j == n_cols - 1:
                i += 1
                j = 0
            else:
                j += 1

        output_figure(plt=f, file_name=file_name)

    def _save_posterior(self, samples, file_name, alpha=0.05, parameter_names=None, significant_digits=None):

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        # first row
        first_row = ['Parameter', 'Mean', 'Credible Interval', 'Confidence Interval']
        rows = [first_row]

        for i in range(len(self.priorRanges)):
            stat = SummaryStat(samples[i])
            mean = format_number(stat.get_mean(), sig_digits=significant_digits)
            credible_interval = format_interval(stat.get_PI(alpha=alpha), sig_digits=significant_digits)
            confidence_interval = format_interval(stat.get_t_CI(alpha=alpha), sig_digits=significant_digits)

            rows.append([
                parameter_names[i],
                mean,
                credible_interval,
                 confidence_interval])

        IO.write_csv(file_name=file_name, rows=rows)


class CalibrationRandomSampling(_Calibration):

    def __init__(self, prior_ranges=None):

        _Calibration.__init__(self, prior_ranges=prior_ranges)
        self.resampledSeeds = []
        self.resamples = [[] for _ in range(len(prior_ranges))]  # Initialize samples for each parameter

    def run(self, log_likelihood_func, num_samples=1000, rng=None):

        self._reset()

        if rng is None:
            rng = np.random.RandomState(1)

        param_samples = []
        for key, prior in self.priorRanges.items():
            # Generate samples uniformly within the prior range
            param_samples.append(
                rng.uniform(low=prior[0], high=prior[1], size=num_samples)
            )

        for i in range(num_samples):
            print('Iteration:', i + 1, '/', num_samples)
            seed = i # rng.randint(0, iinfo(int32).max)

            thetas = [param_samples[j][i] for j in range(len(self.priorRanges))]

            ll, accepted_seed = log_likelihood_func(thetas=thetas, initial_seed=seed)

            if ll != float('-inf'):
                self.seeds.append(accepted_seed)
                self.logLikelihoods.append(ll)
                for i in range(len(self.priorRanges)):
                    self.samples[i].append(thetas[i])

    def resample(self, n_resample=1000, weighted=False):

        if weighted:
            probs = self._get_probs(likelihoods=self.logLikelihoods)

            rng = np.random.RandomState(1)

            # clear the resamples
            self.resampledSeeds.clear()
            for row in self.resamples:
                row.clear()

            sampled_row_indices = rng.choice(
                a=range(0, len(probs)),
                size=n_resample,
                replace=True,
                p=probs)
        else:
            # sort the indices in ascending order of log-likelihoods

            # Pair each number with its original index
            indexed_values = list(enumerate(self.logLikelihoods))

            # Sort by number in decreasing order
            sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)

            # Extract the sorted numbers and their original indices
            sampled_row_indices = [idx for idx, num in sorted_indexed_values]

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        for i in sampled_row_indices:
            self.resampledSeeds.append(self.seeds[i])
            for j in range(len(self.priorRanges)):
                self.resamples[j].append(self.samples[j][i])

    def plot_posterior(self, n_resample=1000, weighted=False, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):

        self.resample(n_resample=n_resample, weighted=weighted)

        self._plot_posterior(
            samples=self.resamples,
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            file_name=file_name,
            parameter_names=parameter_names
        )

    def save_posterior(self, file_name, n_resample=1000, alpha=0.05, parameter_names=None, significant_digits=None):

        self.resample(n_resample=n_resample)

        self._save_posterior(
            samples=self.resamples, file_name=file_name, alpha=alpha,
            parameter_names=parameter_names,
            significant_digits=significant_digits)


class CalibrationMCMCSampling(_Calibration):

    def __init__(self, prior_ranges):

        _Calibration.__init__(self, prior_ranges=prior_ranges)

    def run(self, log_likelihood_func, std_factor=0.1, num_samples=1000, rng=None):
        """Run a simple Metropolis-Hastings MCMC algorithm."""

        self._reset()

        # assert that log_likelihood_func is callable
        if not callable(log_likelihood_func):
            raise ValueError("log_likelihood_func must be a callable function.")
        # assert that log_likelihood_func accepts two parameters: thetas and seed
        sig = inspect.signature(log_likelihood_func)
        if len(sig.parameters) != 2 or 'thetas' not in sig.parameters or 'seed' not in sig.parameters:
            raise ValueError("log_likelihood_func must accept two parameters: thetas and seed.")

        std_factors = [(r[1] - r[0]) * std_factor for k, r in self.priorRanges.items()]

        if rng is None:
            rng = np.random.RandomState(1)

        seed = 0 # rng.randint(0, iinfo(int32).max)

        # Start from a uniform prior
        thetas = np.array(
            [rng.uniform(low=prior_range[0], high=prior_range[1]) for key, prior_range in self.priorRanges.items()])

        log_prior_value = self._log_prior(thetas=thetas)

        log_post = log_prior_value + log_likelihood_func(thetas=thetas, seed=seed)

        for i in range(num_samples):

            thetas_new = rng.normal(thetas, std_factors)
            log_prior = self._log_prior(thetas=thetas_new)
            if log_prior == -np.inf:
                # If the new sample is outside the prior range, skip it
                continue

            log_post_new = (
                    log_prior
                    + log_likelihood_func(thetas=thetas_new, seed=i))

            accept_prob = min(1, np.exp(log_post_new - log_post))

            if rng.random() < accept_prob:
                seed = i
                thetas = thetas_new
                log_post = log_post_new

            if log_post_new != -np.inf:
                self.seeds.append(seed)
                self.logLikelihoods.append(log_post)
                for i in range(len(self.priorRanges)):
                    self.samples[i].append(thetas[i])

    def _log_prior(self, thetas):
        """Compute the log-prior of theta."""

        log_prior = 0
        for range, value in zip(self.priorRanges.values(), thetas):
            if range[0] <= value <= range[1]:
                log_prior += np.log(1 / (range[1] - range[0]))
            else:
                return -np.inf # Outside prior range, log-prior is -inf

        return log_prior

    def plot_posterior(self, n_warmup, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):

        samples = [self.samples[i][n_warmup:] for i in range(len(self.samples))]

        self._plot_posterior(
            samples=samples,
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            file_name=file_name,
            parameter_names=parameter_names
        )

    def save_posterior(self, file_name, n_warmup, alpha=0.05, parameter_names=None, significant_digits=None):

        samples = [self.samples[i][n_warmup:] for i in range(len(self.samples))]

        self._save_posterior(
            samples=samples, file_name=file_name, alpha=alpha, parameter_names=parameter_names,
            significant_digits=significant_digits)
