import numpy

from test_statistical_classes import mytest_ratio_stat_indp

# x_deff = y + increase
# x_ratio = y * ratio
# x_relative_ratio = y * (ration + 1)
Y_MEAN, Y_SD = 10, 4
INCREASE, INCREASE_SD = 3, 1
RATIO, RATIO_SD = 2, 0.5

# generate sample data
y = numpy.random.normal(Y_MEAN, Y_SD, 5000)
increase = numpy.random.normal(INCREASE, INCREASE_SD, 5000)
ratio = numpy.random.normal(RATIO, RATIO_SD, 5000)

x_diff = y + increase
x_ratio = numpy.multiply(y, ratio)
x_relative_ratio = numpy.multiply(y, ratio + 1)

# test statistics for the relative difference of two paired samples
mytest_ratio_stat_indp(x=x_ratio, y=y,
                       expected_value=RATIO, st_dev='Unknown')
