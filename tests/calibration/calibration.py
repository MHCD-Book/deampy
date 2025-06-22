import numpy as np
import scipy.stats as stats

from deampy.calibration import CalibrationMCMCSampling, CalibrationRandomSampling

SIM_OBSS = (9.0, -3.0)  # Observed data point
PRIOR_RANGES = (
    (0.0, 20.0),  # Uniform prior range for theta[0]
    (-5.0, 5.0)   # Uniform prior range for theta[1]
)


def simulate(thetas, seed):
    """Simulate data from a model with parameter theta."""

    rng = np.random.RandomState(seed)

    return [thetas[0] + rng.normal(0, 1, size=1)[0], thetas[1] + rng.normal(0, 1, size=1)[0]]

def log_likelihood(thetas, seed):
    """Compute the log-likelihood of observed data given theta."""

    sim_output = simulate(thetas=thetas, seed=seed)  # Simulate data

    if not isinstance(sim_output, list):
        sim_output = [sim_output]

    weight = 0
    for i, x_obs in enumerate(SIM_OBSS):
        weight += stats.norm.logpdf(
            x=x_obs,
            loc=sim_output[i],
            scale=1)

    return weight


if __name__ == "__main__":

    # Run random sampling calibration with the specified prior ranges and log-likelihood function
    random_sampling = CalibrationRandomSampling(prior_ranges=PRIOR_RANGES)
    random_sampling.run(log_likelihood_func=log_likelihood, num_samples=5000)
    random_sampling.save(file_name="random_sampling.csv")
    random_sampling.save_posterior(file_name="random_sampling_posterior.csv", n_resample=1000)

    random_sampling.plot_posterior(
        n_resample=1000, n_cols=2, n_rows=1, figsize=(10, 5),
        file_name='figs/random_sampling_posterior_plot.png')

    # Run MCMC calibration with the specified prior ranges and log-likelihood function
    mcmc = CalibrationMCMCSampling(prior_ranges=PRIOR_RANGES, std_factor=0.05)
    mcmc.run(log_likelihood_func=log_likelihood, num_samples=5000)

    # Save the MCMC results
    mcmc.plot_trace(n_cols=2, n_rows=1, figsize=(10, 5), share_x=True,
                    file_name='figs/mcmc_trace_plot.png', moving_ave_window=1000)

    mcmc.plot_posterior(n_warmup=2000, n_cols=2, n_rows=1, figsize=(10, 5),
                        file_name='figs/mcmc_posterior_plot.png')

