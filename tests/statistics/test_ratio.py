import numpy

from test_statistical_classes import (mytest_ratio_stat_indp, mytest_ratio_of_means_stat_paired,
                                      mytest_relative_diff_stat_indp)

# x_deff = y + increase
# x_ratio = y * ratio
# x_relative_ratio = y * (ration + 1)
Y_MEAN, Y_SD = 10, 0.4
INCREASE, INCREASE_SD = 3, 1
RATIO, RATIO_SD = 2, 0.5

# generate sample data
numpy.random.seed(1)
y = numpy.random.normal(Y_MEAN, Y_SD, 5000)
increase = numpy.random.normal(INCREASE, INCREASE_SD, 5000)
ratio = numpy.random.normal(RATIO, RATIO_SD, 5000)

x_diff = y + increase
x_ratio = numpy.multiply(y, ratio)
x_relative_ratio = numpy.multiply(y, ratio + 1)

print('average of x_ratio:', numpy.mean(x_ratio))

# test statistics for the relative difference of two paired samples
mytest_ratio_stat_indp(x=x_ratio, y=y,
                       expected_value='Unknown', st_dev='Unknown')

# test statistics for the relative difference of two independent samples
mytest_relative_diff_stat_indp(x=x_relative_ratio, y=y,
                               expected_value='Unknown', st_dev='Unknown')

mytest_ratio_of_means_stat_paired(x=x_diff, y=y,
                                  expected_value='Unknown', st_dev='Unknown')
