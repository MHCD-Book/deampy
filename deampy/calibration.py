import inspect

import matplotlib.pyplot as plt
import numpy as np
from numpy import iinfo, int32
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

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

        self.samples = {}
        self.seeds = None
        self.logLikelihoods = None

        self.priorRanges = prior_ranges  # List of tuples (min, max) for each parameter
        self.ifABC = False
        self._reset()

    def _reset(self):
        """Reset the calibration object."""
        self.samples = {key: [] for key in self.priorRanges} # Initialize samples for each parameter
        self.seeds = []
        self.logLikelihoods = []

    def run(self, *args, **kwargs):
        """Run the calibration method."""
        raise NotImplementedError("Subclasses should implement this method.")

    def save_samples(self, file_name):

        cols = {
            'Seed': self.seeds,
            'Log-Likelihood': self.logLikelihoods
        }
        # add parameter samples to the columns
        for key in self.priorRanges:
            cols[key] = self.samples[key]

        # write the calibration result into a csv file
        IO.write_dictionary_to_csv(
            dictionary=cols,
            file_name=file_name)

    def read_samples(self, file_name):
        """Read samples from a CSV file."""

        self._reset()

        cols = IO.read_csv_cols_to_dictionary(file_name=file_name, if_convert_float=True)

        # the first column is seeds
        self.seeds = cols['Seed'].astype(int).tolist()
        # the second column is log-likelihoods
        self.logLikelihoods = cols['Log-Likelihood'].tolist()
        # remaining columns are parameter samples
        for key in self.priorRanges:
            self.samples[key] = cols[key].tolist()

    def _error_check_log_func(self, log_likelihood_func):
        """Check if the log_likelihood_func is callable and has the correct signature."""

        # Ensure that the log_likelihood_func is callable and has the correct signature
        if not callable(log_likelihood_func):
            raise ValueError("log_likelihood_func must be a callable function.")
        sig = inspect.signature(log_likelihood_func)
        if 'thetas' not in sig.parameters or 'seed' not in sig.parameters:
            raise ValueError("log_likelihood_func must accept two parameters: thetas and seed.")
        if 'epsilon_ll' in sig.parameters:
            self.ifABC = True

    def _record_itr(self, ll, thetas, accepted_seed):
        """Record the iteration results if the log-likelihood is not -inf."""

        self.seeds.append(accepted_seed)
        self.logLikelihoods.append(ll)
        i = 0
        for key in self.priorRanges:
            self.samples[key].append(thetas[i])
            i += 1

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
        for key in parameter_names:

            range = self.priorRanges[key]

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
                    samples=self.samples[key],
                    par_name=key,
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

    def _plot_posteriors(self, samples, n_rows=1, n_cols=1, figsize=(7, 5),
                         file_name=None, parameter_names=None):
        """Plot the posterior distribution of the MCMC samples."""

        # plot each panel
        f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

        if parameter_names is None:
            parameter_names = self.priorRanges.keys()

        i = 0
        j = 0
        for key in parameter_names:

            range = self.priorRanges[key]

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
                    data=samples[key],
                    # color='blue',
                    title=key,
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

    def _plot_pairwise_posteriors(
            self, samples,figsize=(7, 7), file_name=None, parameter_names=None):
        """Plot pairwise posterior distributions."""

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        # plot each panel
        n = len(parameter_names)

        f, axarr = plt.subplots(nrows=n, ncols=n, figsize=figsize)

        for i in range(n):
            for j in range(n):

                # get the current axis
                ax = axarr[i, j]

                # plot histogram for diagonal elements
                if i == j:
                    # plot histogram
                    add_histogram_to_ax(
                        ax=ax,
                        data=samples[parameter_names[i]],
                        x_range=self.priorRanges[parameter_names[i]],
                    )
                    # ax.set_yticklabels([])
                    # ax.set_yticks([])

                else:

                    x_data = np.array(samples[parameter_names[i]])
                    y_data = np.array(samples[parameter_names[j]])

                    ax.scatter(x_data,
                               y_data,
                               alpha=0.5, s=2)
                    ax.set_xlim(self.priorRanges[parameter_names[i]])
                    ax.set_ylim(self.priorRanges[parameter_names[j]])
                    # correlation line
                    b, m = polyfit(x_data, y_data, 1)
                    ax.plot(x_data, b + m * x_data, '-', c='black')
                    corr, p = pearsonr(x_data, y_data)
                    ax.text(0.95, 0.95, '{0:.2f}'.format(corr), transform=ax.transAxes, fontsize=6,
                            va='top', ha='right')

                if j == 0:
                    ax.set_ylabel(parameter_names[i])
                if i == n - 1:
                    ax.set_xlabel(parameter_names[j])

        f.align_ylabels(axarr[:, 0])
        f.tight_layout()
        output_figure(plt=f, file_name=file_name, dpi=300)

    def _save_posteriors(self, samples, file_name, alpha=0.05, parameter_names=None, significant_digits=None):

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        # first row
        first_row = ['Parameter', 'Mean', 'Credible Interval', 'Confidence Interval']
        rows = [first_row]

        for key in parameter_names:
            stat = SummaryStat(samples[key])
            mean = format_number(stat.get_mean(), sig_digits=significant_digits)
            credible_interval = format_interval(stat.get_PI(alpha=alpha), sig_digits=significant_digits)
            confidence_interval = format_interval(stat.get_t_CI(alpha=alpha), sig_digits=significant_digits)

            rows.append([
                key,
                mean,
                credible_interval,
                 confidence_interval])

        IO.write_csv(file_name=file_name, rows=rows)


class CalibrationRandomSampling(_Calibration):

    def __init__(self, prior_ranges=None):

        _Calibration.__init__(self, prior_ranges=prior_ranges)
        self.resampledSeeds = []
        self.resamples = {key: [] for key in prior_ranges}  # Initialize samples for each parameter

    def run(self, log_likelihood_func, num_samples=1000, rng=None, print_iterations=True):

        self._error_check_log_func(log_likelihood_func)
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

            seed = i # rng.randint(0, iinfo(int32).max)

            thetas = [param_samples[j][i] for j in range(len(self.priorRanges))]

            output = log_likelihood_func(thetas=thetas, seed=seed)
            if isinstance(output, tuple):
                ll, accepted_seed = output
            else:
                ll = output
                accepted_seed = seed

            if print_iterations:
                print('Iteration: {}/{} | Log-Likelihood: {}'.format(i + 1, num_samples, ll))

            if ll != -np.inf:
                self._record_itr(ll=ll, thetas=thetas, accepted_seed=accepted_seed)


    def resample(self, n_resample=1000, weighted=False):

        if weighted:
            probs = self._get_probs(likelihoods=self.logLikelihoods)

            rng = np.random.RandomState(1)

            # clear the resamples
            self.resampledSeeds.clear()
            for key, row in self.resamples.items():
                row.clear()

            sampled_row_indices = rng.choice(
                a=range(0, len(probs)),
                size=n_resample,
                replace=True,
                p=probs)
        else:

            if n_resample > len(self.logLikelihoods):
                raise ValueError("n_resample cannot be greater than the number of samples.")

            # sort the indices in ascending order of log-likelihoods
            # Pair each number with its original index
            indexed_values = list(enumerate(self.logLikelihoods))

            # Sort by number in decreasing order
            sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)

            # Extract the sorted numbers and their original indices
            sampled_row_indices = [idx for idx, num in sorted_indexed_values]

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        self.resampledSeeds.clear()
        self.resamples = {key: [] for key in self.priorRanges}  # Reset resamples
        for i in range(n_resample):

            row_index = sampled_row_indices[i]
            self.resampledSeeds.append(self.seeds[row_index])
            for key in self.priorRanges:
                self.resamples[key].append(self.samples[key][row_index])

    def plot_posterior(self, n_resample=1000, weighted=False, n_rows=1, n_cols=1, figsize=(7, 5),
                       file_name=None, parameter_names=None):

        self.resample(n_resample=n_resample, weighted=weighted)

        self._plot_posteriors(
            samples=self.resamples,
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            file_name=file_name,
            parameter_names=parameter_names
        )

    def plot_pairwise_posteriors(self, n_resample=1000, weighted=False,
                                 figsize=(7, 7), file_name=None, parameter_names=None):

        self.resample(n_resample=n_resample, weighted=weighted)

        self._plot_pairwise_posteriors(
            samples=self.resamples,
            figsize=figsize,
            file_name=file_name,
            parameter_names=parameter_names
        )


    def save_posterior(self, file_name, n_resample=1000,
                       alpha=0.05, weighted=False, parameter_names=None, significant_digits=None):

        self.resample(n_resample=n_resample, weighted=weighted)

        self._save_posteriors(
            samples=self.resamples, file_name=file_name, alpha=alpha,
            parameter_names=parameter_names,
            significant_digits=significant_digits)


class CalibrationMCMCSampling(_Calibration):

    def __init__(self, prior_ranges):

        _Calibration.__init__(self, prior_ranges=prior_ranges)

    def _get_ll(self, log_likelihood_func, thetas, seed, epsilon_ll=-np.inf):

        if not self.ifABC:
            output = log_likelihood_func(thetas=thetas, seed=seed)
        else:
            output = log_likelihood_func(thetas=thetas, seed=seed, epsilon_ll=epsilon_ll)

        return output

    def run(self, log_likelihood_func, std_factor=0.1, epsilon_ll=-np.inf, num_samples=1000, rng=None, print_iterations=True):
        """Run a simple Metropolis-Hastings MCMC algorithm."""

        self._error_check_log_func(log_likelihood_func)
        self._reset()

        if rng is None:
            rng = np.random.RandomState(1)

        # std factors for each parameter based on the prior ranges
        std_factors = [(r[1] - r[0]) * std_factor for k, r in self.priorRanges.items()]

        # initial parameter samples from uniform priors
        thetas = np.array(
            [rng.uniform(low=prior_range[0], high=prior_range[1]) for key, prior_range in self.priorRanges.items()])

        # generate a random seed for the first sample
        seed = rng.randint(0, iinfo(int32).max)

        # compute the log-prior and log-posterior for the initial sample
        log_prior_value = self._log_prior(thetas=thetas)
        output = self._get_ll(
            log_likelihood_func=log_likelihood_func, thetas=thetas, seed=seed, epsilon_ll=epsilon_ll)

        if isinstance(output, tuple):
            ll, accepted_seed = output
        else:
            ll = output
            accepted_seed = seed
        log_post = log_prior_value + ll

        for i in range(num_samples):

            thetas_new = rng.normal(thetas, std_factors)
            log_prior = self._log_prior(thetas=thetas_new)
            if log_prior == -np.inf:
                # If the new sample is outside the prior range, skip it
                continue

            seed = rng.randint(0, iinfo(int32).max)

            output = self._get_ll(
                log_likelihood_func=log_likelihood_func, thetas=thetas_new, seed=seed, epsilon_ll=epsilon_ll)

            if isinstance(output, tuple):
                ll, accepted_seed = output
            else:
                ll = output
                accepted_seed = seed

            log_post_new = log_prior + ll

            accept_prob = min(1, np.exp(log_post_new - log_post))

            if rng.random() < accept_prob:
                # seed = accepted_seed
                thetas = thetas_new
                log_post = log_post_new

            if print_iterations:
                print('Iteration: {}/{} | Log-Likelihood: {}'.format(i + 1, num_samples, log_post))

            self._record_itr(ll=log_post, thetas=thetas, accepted_seed=accepted_seed)

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

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        samples = {key: self.samples[key][n_warmup:] for key in parameter_names}

        self._plot_posteriors(
            samples=samples,
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            file_name=file_name,
            parameter_names=parameter_names
        )

    def save_posterior(self, file_name, n_warmup, alpha=0.05, parameter_names=None, significant_digits=None):

        if parameter_names is None:
            parameter_names = list(self.priorRanges.keys())

        samples = {key: self.samples[key][n_warmup:] for key in parameter_names}

        self._save_posteriors(
            samples=samples, file_name=file_name, alpha=alpha, parameter_names=parameter_names,
            significant_digits=significant_digits)
