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

        assert isinstance(prior_ranges, (list, tuple)) and len(prior_ranges) > 0, \
            "prior_ranges must be a non-empty list of tuples (min, max) for each parameter."
        # if prior_ranges is a list of two numbers, convert it to a list of one tuple
        if len(prior_ranges) == 2 and isinstance(prior_ranges[0], (int, float)) and isinstance(prior_ranges[1], (int, float)):
            prior_ranges = [prior_ranges]

        self.priorRanges = prior_ranges
        self.samples = [[] for i in range(len(prior_ranges))]  # Initialize samples for each parameter
        self.seeds = []
        self.logLikelihoods = []
        self.probs = []  # Normalized probabilities for each sample

    def run(self, *args, **kwargs):
        """Run the calibration method."""
        raise NotImplementedError("Subclasses should implement this method.")

    def save_samples(self, file_name, parameter_names=None):

        if parameter_names is None:
            parameter_names = [f'Parameter {i+1}' for i in range(len(self.priorRanges))]

        # first row
        if isinstance(self, CalibrationRandomSampling):
            first_row = ['Seed', 'Log-Likelihood', 'Probabilities']
        elif isinstance(self, CalibrationMCMCSampling):
            first_row = ['Seed', 'Log-Likelihood']
        else:
            raise ValueError("Unknown calibration method")
        first_row.extend(parameter_names)

        # produce the list to report the results
        csv_rows = [first_row]

        for i in range(len(self.seeds)):

            if isinstance(self, CalibrationRandomSampling):
                row = [self.seeds[i], self.logLikelihoods[i], self.probs[i]]
            elif isinstance(self, CalibrationMCMCSampling):
                row = [self.seeds[i], self.logLikelihoods[i]]
            else:
                raise ValueError("Unknown calibration method")

            row.extend([self.samples[j][i] for j in range(len(self.priorRanges))])

            csv_rows.append(row)

        # write the calibration result into a csv file
        IO.write_csv(
            file_name=file_name,
            rows=csv_rows)

    def read_samples(self, file_name):
        """Read samples from a CSV file."""
        cols = IO.read_csv_cols(file_name=file_name, if_ignore_first_row=True, if_convert_float=True)

        # first column is seeds
        self.seeds = cols[0].astype(int).tolist()
        # second column is log-likelihoods
        self.logLikelihoods = cols[1].tolist()
        k = 2  # start from the third column
        if isinstance(self, CalibrationRandomSampling):
            # third column is probabilities
            self.probs = cols[k].tolist()
            k += 1  # move to the next column

        # remaining columns are parameter samples
        for i in range(len(self.priorRanges)):
            self.samples[i].extend(cols[i + k].tolist())

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

    def _plot_posterior(self, samples, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):
        """Plot the posterior distribution of the MCMC samples."""

        # plot each panel
        f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

        if parameter_names is None:
            parameter_names = [f'Parameter {i+1}' for i in range(len(self.priorRanges))]

        for i in range(n_rows):
            for j in range(n_cols):
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
                        data=samples[i*n_cols+j],  # Skip warmup samples
                        # color='blue',
                        title=parameter_names[i * n_cols + j],
                        x_label='Sampled Values',
                        # y_label=None,
                        x_range=self.priorRanges[i * n_cols + j],
                        y_range=None,
                        transparency=0.7,
                    )

        output_figure(plt=f, file_name=file_name)


    def _save_posterior(self, samples, file_name, alpha=0.05, parameter_names=None, significant_digits=None):

        if parameter_names is None:
            parameter_names = [f'Parameter {i+1}' for i in range(len(self.priorRanges))]

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

    def __init__(self, prior_ranges):

        _Calibration.__init__(self, prior_ranges=prior_ranges)
        self.resampledSeeds = []
        self.resamples = [[] for _ in range(len(prior_ranges))]  # Initialize samples for each parameter

    def run(self, log_likelihood_func, num_samples=1000, rng=None):

        if rng is None:
            rng = np.random.RandomState(1)

        param_samples = []
        for prior in self.priorRanges:
            # Generate samples uniformly within the prior range
            param_samples.append(
                rng.uniform(low=prior[0], high=prior[1], size=num_samples)
            )

        for i in range(num_samples):

            seed = i # rng.randint(0, iinfo(int32).max)

            thetas = [param_samples[j][i] for j in range(len(self.priorRanges))]

            ll = log_likelihood_func(thetas=thetas, seed=seed)

            self.seeds.append(seed)
            self.logLikelihoods.append(ll)
            for i in range(len(self.priorRanges)):
                self.samples[i].append(thetas[i])

        self.probs = self._get_probs(likelihoods=self.logLikelihoods)

    def resample(self, n_resample=1000):

        rng = np.random.RandomState(1)

        # clear the resamples
        self.resampledSeeds.clear()
        for row in self.resamples:
            row.clear()

        sampled_row_indices = rng.choice(
            a=range(0, len(self.probs)),
            size=n_resample,
            replace=True,
            p=self.probs)

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        for i in sampled_row_indices:
            self.resampledSeeds.append(self.seeds[i])
            for j in range(len(self.priorRanges)):
                self.resamples[j].append(self.samples[j][i])

    def plot_posterior(self, n_resample=1000, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):

        self.resample(n_resample=n_resample)

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
            samples=self.resamples, file_name=file_name, alpha=alpha, parameter_names=parameter_names,
            significant_digits=significant_digits)


class CalibrationMCMCSampling(_Calibration):

    def __init__(self, prior_ranges):

        _Calibration.__init__(self, prior_ranges=prior_ranges)

    def run(self, log_likelihood_func, std_factor=0.1, num_samples=1000, rng=None):
        """Run a simple Metropolis-Hastings MCMC algorithm."""

        # assert that log_likelihood_func is callable
        if not callable(log_likelihood_func):
            raise ValueError("log_likelihood_func must be a callable function.")
        # assert that log_likelihood_func accepts two parameters: thetas and seed
        sig = inspect.signature(log_likelihood_func)
        if len(sig.parameters) != 2 or 'thetas' not in sig.parameters or 'seed' not in sig.parameters:
            raise ValueError("log_likelihood_func must accept two parameters: thetas and seed.")

        std_factors = [(r[1] - r[0]) * std_factor for r in self.priorRanges]

        if rng is None:
            rng = np.random.RandomState(1)

        seed = 0 # rng.randint(0, iinfo(int32).max)

        # Start from a uniform prior
        thetas = np.array(
            [rng.uniform(low=prior_range[0], high=prior_range[1]) for prior_range in self.priorRanges])

        log_prior_value = self._log_prior(thetas=thetas)

        log_post = log_prior_value + log_likelihood_func(thetas=thetas, seed=seed)

        for i in range(num_samples):

            seed = i # rng.randint(0, iinfo(int32).max)

            thetas_new = rng.normal(thetas, std_factors)
            log_post_new = (
                    self._log_prior(thetas=thetas_new)
                    + log_likelihood_func(thetas=thetas_new, seed=seed))

            accept_prob = min(1, np.exp(log_post_new - log_post))

            if rng.random() < accept_prob:
                thetas = thetas_new
                log_post = log_post_new

            self.seeds.append(seed)
            self.logLikelihoods.append(log_post)
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

    def save_posterior(self, file_name, n_warmup, alpha=0.05, parameter_names=None):

        samples = [self.samples[i][n_warmup:] for i in range(len(self.samples))]

        self._save_posterior(samples=samples, file_name=file_name, alpha=alpha, parameter_names=parameter_names)
