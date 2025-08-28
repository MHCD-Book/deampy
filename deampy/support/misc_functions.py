import random

import numpy as np
from scipy.stats import norm


def proper_file_name(text):
    """
    :param text: file name
    :return: filename where invalid characters are removed
    """

    return text.replace('|', ',').replace('<', 'l').replace('>', 'g').replace('\n', '')


def get_cumulative_mean(data):
    """
    calculates the cumulative mean of a time-series
    :param data: list of observations
    :return: list of cumulative means
    """

    cum_means = []
    total = 0
    n = 0
    for i in range(len(data)):
        if data[i] is not None:
            total += data[i]
            n += 1
        if n > 0:
            cum_means.append(total/n)
        else:
            cum_means.append(None)

    return cum_means


def get_cumulative_var(data):
    """
    calculates the cumulative variance of a time-series
    :param data: list of observations
    :return: list of cumulative variances
    """

    cum_vars = []
    total = 0
    total_sq = 0
    n = 0
    for i in range(len(data)):
        if data[i] is not None:
            total += data[i]
            total_sq += data[i]**2
            n += 1
        if n > 1:
            cum_vars.append((total_sq - (total**2)/n)/(n-1))
        else:
            cum_vars.append(None)

    return cum_vars

def get_moving_average(data, window=2):
    """
    calculates the moving average of a time-series
    :param data: list of observations
    :param window: the window (number of data points) over which the average should be calculated
    :return: list of moving averages
    """

    if window < 2:
        raise ValueError('The window over which the moving averages '
                         'should be calculated should be greater than 1.')
    if window >= len(data):
        window = len(data) - 1
        # raise ValueError('The window over which the moving averages '
        #                  'should be calculated should be less than the number of data points.')

    window = int(window)

    averages = []

    # the 'window - 1' averages cannot be calculated
    for i in range(window-1):
        averages.append(None)

    # calculate the first moving average
    total = 0
    n = 0
    for i in range(window):
        if data[i] is not None:
            total += data[i]
            n += 1
    moving_ave = total/n
    averages.append(moving_ave)

    for i in range(window, len(data)):
        if moving_ave is None:
            moving_sum = 0
        else:
            moving_sum = moving_ave * n
        if data[i-window] is not None:
            moving_sum -= data[i-window]
            n -= 1
        if data[i] is not None:
            moving_sum += data[i]
            n += 1

        if n > 0:
            moving_ave = moving_sum/n
        else:
            moving_ave = None
        averages.append(moving_ave)

    return averages


def effective_sample_size(likelihood_weights):
    """
    :param likelihood_weights: (list) probabilities
    :returns: the effective sample size
    """

    # convert to numpy array if needed
    if not type(likelihood_weights) == np.ndarray:
        likelihood_weights = np.array(likelihood_weights)

    # normalize the weights if needed
    s = sum(likelihood_weights)
    if s < 0.99999 or s > 1.00001:
        likelihood_weights = likelihood_weights / s

    return 1 / np.sum(likelihood_weights ** 2)


def get_random_colors(n, seed=0):
    """
    :param n: (int) number of colors randomly selected 
    :param seed: (int) seed of the random number generator 
    :return: (list) of color codes 
    """
    
    random.seed(seed)

    colors = []
    for j in range(n):
        rand_colors = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        colors.append(rand_colors)
    
    return colors


def convert_lnl_to_prob(ln_likelihoods):

    for i, lnl in enumerate(ln_likelihoods):
        if np.isnan(lnl):
            ln_likelihoods[i] = -np.inf

    # find the maximum lnl
    max_lnl = max(ln_likelihoods)

    # find probability weights
    probs = [np.exp(s - max_lnl) for s in ln_likelihoods]
    sum_prob = sum(probs)

    # normalize the probability weights
    return np.array(probs)/sum_prob


def get_prob_normal_greater_0(mean, st_dev):
    """
    :param mean: (float) mean of the normal distribution
    :param st_dev: (float) standard deviation of the normal distribution
    :return: the probability of the random variable following this normal distribution is greater than 0
    """

    return 1 - norm(loc=mean, scale=st_dev).cdf(0)


def get_prob_x_greater_than_y(x_mean, x_st_dev, y_mean, y_st_dev, corr=0):
    """
    :param x_mean: (float) the mean of x, which follows a normal distribution.
    :param x_st_dev: (float) the standard deviation of x
    :param y_mean: (float)the mean of y, which follows a normal distribution
    :param y_st_dev: (float) the standard deviation of y
    :param corr: (float) the correlation between x and y
    :return: the probability that x > y
    """

    return get_prob_normal_greater_0(
        mean=x_mean - y_mean, st_dev=np.sqrt(x_st_dev**2 + y_st_dev**2 + 2*corr*x_st_dev*y_st_dev))


def get_prob_x_greater_than_ys(x_mean, x_st_dev, y_means, y_st_devs):
    """
    :param x_mean: (float) the mean of x, which follows a normal distribution
    :param x_st_dev: (float) the standard deviation of x
    :param y_means: (list) the means of ys, which follow normal distributions
    :param y_st_devs: (list) the standard deviations of ys
    :return: the probability that x is greater than all ys (i.e., x > y_1 and x > y_2, and ...)
        (note all distributions should be independent)
    """

    prob = 1
    for i in range(len(y_means)):
        prob *= get_prob_normal_greater_0(
            mean=x_mean - y_means[i],
            st_dev=np.sqrt(x_st_dev**2 + y_st_devs[i]**2))

    return prob


if __name__ == "__main__":

    print(get_prob_normal_greater_0(-1, 1))
    print(get_prob_x_greater_than_y(x_mean=10, x_st_dev=2,
                                    y_mean=9, y_st_dev=2))
    print(get_prob_x_greater_than_ys(x_mean=10, x_st_dev=2,
                                     y_means=[9, 11], y_st_devs=[2, 2]))

    print(get_cumulative_mean([1, 2, 3]))
    print(get_cumulative_var([1, 2, 3]))