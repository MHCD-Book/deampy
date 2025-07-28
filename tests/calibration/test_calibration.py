import numpy as np
import scipy.stats as stats

from deampy.calibration import CalibrationMCMCSampling, CalibrationRandomSampling

SIM_OBSS = (9.0, -3.0)  # Observed data point
PRIOR_RANGES = {
    'Par 1': (0.0, 20.0),  # Uniform prior range for theta[0]
    'Par 2': (-5.0, 5.0)   # Uniform prior range for theta[1]
}
N_SAMPLES = 5000  # Number of samples for random sampling
N_RESAMPLES = 1000

ST_FACTOR = 0.1
WARM_UP = 2000  # Number of warm-up iterations for MCMC
EPSILON_LL = -10

def simulate(thetas, seed):
    """Simulate data from a model with parameter theta."""

    rng = np.random.RandomState(seed)

    return [thetas[0] + rng.normal(0, 1, size=1)[0], thetas[1] + rng.normal(0, 1, size=1)[0]]


def log_likelihood_func(thetas, seed):
    """Compute the log-likelihood of observed data given theta."""

    sim_output = simulate(thetas=thetas, seed=seed)  # Simulate data

    if not isinstance(sim_output, list):
        sim_output = [sim_output]

    ll = 0
    for i, x_obs in enumerate(SIM_OBSS):
        ll += stats.norm.logpdf(
            x=x_obs,
            loc=sim_output[i],
            scale=1)

    return ll


def binary_log_likelihood_func(thetas, seed, epsilon_ll):
    """Compute the log-likelihood of observed data given theta."""

    ll = log_likelihood_func(thetas=thetas, seed=seed)

    if ll > epsilon_ll:
        ll = 0
    else:
        ll = -np.inf
    return ll


def test_ramdom_sampling():

    # Run random sampling calibration with the specified prior ranges and log-likelihood function
    random_sampling = CalibrationRandomSampling(prior_ranges=PRIOR_RANGES)
    random_sampling.run(log_likelihood_func=log_likelihood_func, num_samples=N_SAMPLES)
    random_sampling.save_samples(file_name="output/rnd_sampling.csv")

    for weighted in [True, False]:

        text  = "weighted" if weighted else "unweighted"

        # save results based on weighted resampling
        random_sampling.save_posterior(
            file_name="output/rnd_sampling_postr_{}.csv".format(text),
            n_resample=N_RESAMPLES, weighted=weighted)

        random_sampling.plot_posterior(
            n_resample=N_RESAMPLES, weighted=weighted,
            n_cols=2, n_rows=1, figsize=(10, 5),
            file_name='figs/rnd_sampling_postr_{}.png'.format(text))

        random_sampling.plot_pairwise_posteriors(
            n_resample=N_RESAMPLES, weighted=weighted,
            figsize=(10, 10),
            file_name='figs/rnd_sampling_pairwise_postrs_{}.png'.format(text))


def test_mcmc_sampling(log_binary=False):

    # Run MCMC calibration with the specified prior ranges and log-likelihood function
    mcmc = CalibrationMCMCSampling(prior_ranges=PRIOR_RANGES)
    if log_binary:
        mcmc.run(log_likelihood_func=binary_log_likelihood_func,
                 std_factor=ST_FACTOR, epsilon_ll=EPSILON_LL, num_samples=N_SAMPLES)
    else:
        mcmc.run(log_likelihood_func=log_likelihood_func,
                 std_factor=ST_FACTOR, num_samples=N_SAMPLES)


    text = "bin" if log_binary else "approx"

    mcmc.save_samples(file_name="output/mcmc_{}.csv".format(text))
    mcmc.save_posterior(file_name="output/mcmc_postr_{}.csv".format(text), n_warmup=WARM_UP)

    # Save the MCMC results
    mcmc.plot_trace(n_cols=2, n_rows=1, figsize=(10, 5), share_x=True,
                    file_name='figs/mcmc_trace_plot_{}.png'.format(text), moving_ave_window=1000)

    mcmc.plot_posterior(
        n_warmup=WARM_UP, n_cols=2, n_rows=1, figsize=(10, 5),
        file_name='figs/mcmc_postr_plot_{}.png'.format(text))


if __name__ == "__main__":
    #
    # test_ramdom_sampling()
    test_mcmc_sampling(log_binary=False)
    test_mcmc_sampling(log_binary=True)




