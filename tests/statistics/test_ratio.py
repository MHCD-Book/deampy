import numpy as np

from test_statistical_classes import *

Y_MEAN, Y_SD = 10, 0.4        # baseline (Y_ref)
INCREASE, INCREASE_SD = 3, 1  # X will be Y + INCREASE
RATIO, RATIO_SD = 2, 0.5      # X will be Y * RATIO

# generate realizations for Y
np.random.seed(1)
y = np.random.normal(Y_MEAN, Y_SD, 5000)

# generate realizations for increase and ratio
increase = np.random.normal(INCREASE, INCREASE_SD, 5000)
ratio = np.random.normal(RATIO, RATIO_SD, 5000)

# generate realizations for X
x_diff = y + increase
x_ratio = np.multiply(y, ratio)
x_relative_ratio = np.multiply(y, ratio + 1)

# report averages
print('average of y:', np.mean(y))
print('average of x_diff:', np.mean(x_diff))
print('average of x_ratio:', np.mean(x_ratio))
print('average of x_relative_ratio:', np.mean(x_relative_ratio))

# ratio under independent sampling
mytest_ratio_stat_indp(
    x=x_ratio, y=y, expected_value=RATIO, st_dev='Unknown')
# ratio under paired sampling
mytest_ratio_stat_paired(
    x=x_ratio, y=y, expected_value=RATIO, st_dev='Unknown')

# ratio of means under independent sampling
mytest_ratio_of_means_stat_indp(
    x=x_ratio, y=y, expected_value=RATIO, st_dev='Unknown')
# ratio of means under paired sampling
mytest_ratio_of_means_stat_paired(
    x=x_ratio, y=y, expected_value=RATIO, st_dev='Unknown')


# relative difference under independent sampling
mytest_relative_diff_stat_indp(x=x_relative_ratio, y=y,
                               expected_value='1.3', st_dev='Unknown')

# relative difference under paired sampling
mytest_ratio_of_means_stat_paired(x=x_diff, y=y,
                                  expected_value='1.3', st_dev='Unknown')
