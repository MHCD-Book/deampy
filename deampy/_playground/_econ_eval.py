import numpy as np
import scipy.stats as stat
from statsmodels.stats.power import TTestPower


def get_min_monte_carlo_samples_power_calc(delta_costs, delta_effects, hypothesized_true_wtp, alpha=0.05, power=0.8):
    """
    :param delta_costs: (list) of marginal cost observations
    :param delta_effects: (list) of marginal effect observations
    :param hypothesized_true_wtp: (double) the hypothesized true WTP threshold at which the NMB lines cross
    :param alpha: (double) significance level
    :param power: (double) between (0, 1) the desired statistical power
    :return: (int) the minimum Monte Carlo samples needed based on power calculation
    """

    diff = delta_costs - hypothesized_true_wtp * delta_effects
    standardized_diff = np.average(diff)/np.std(diff)

    # Initiate the power analysis
    power_analysis = TTestPower()
    # Calculate sample size
    sample_size = power_analysis.solve_power(
        effect_size=standardized_diff, alpha=alpha, power=power, alternative='two-sided')

    # round the estimated number of required Monte Carlo samples and increment it by 1
    return round(sample_size) + 1


def get_prob_minus_power(n, power, true_wtp, wtp_error, mean_d_cost, mean_d_effect, var_plus_error, var_minus_error):
    """
    :param n: (int) the number of Monte Carlo samples
    :param power: (double) between (0, 1), the minimum value we want
        Pr{|true_wtp - estimated_wtp| < wtp_error} to be
    :param true_wtp: (double) true wtp value
    :param wtp_error: the error in estimating the true WTP value at which NMB lines of two alternatives intersect
    :param mean_d_cost: (double) mean of marginal cost
    :param mean_d_effect: (double) mean of marginal effect
    :param var_plus_error: (double) variance of marginal NMB if true intersecting WTP is
        true_wtp_intersection + wtp_error
    :param var_minus_error: (double) variance of marginal NMB if true intersecting WTP is
        true_wtp_intersection - wtp_error
    :return: Pr{|true_wtp - estimated_wtp| < true_wtp * wtp_percent_error} - power for the given n
    """

    prob_plus_error_less_than_0 = stat.norm.cdf(
        x=0, loc=mean_d_cost - (true_wtp + wtp_error) * mean_d_effect,
        scale=np.sqrt(var_plus_error/n))

    prob_minus_error_less_then_0 = stat.norm.cdf(
        x=0, loc=mean_d_cost - (true_wtp - wtp_error) * mean_d_effect,
        scale=np.sqrt(var_minus_error/n))

    if mean_d_effect > 0:
        return (prob_plus_error_less_than_0 - prob_minus_error_less_then_0) - power
    else:
        return -(prob_plus_error_less_than_0 - prob_minus_error_less_then_0) - power
