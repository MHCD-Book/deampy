import numpy

from deampy import format_functions as form
from deampy import statistics as stat


def print_results(stat):
    print('   Average =', form.format_number(stat.get_mean(), deci=3))
    print('   St Dev =', form.format_number(stat.get_stdev(), deci=3))
    print('   Min =', form.format_number(stat.get_min(), deci=3))
    print('   Max =', form.format_number(stat.get_max(), deci=3))
    print('   Median =', form.format_number(stat.get_percentile(50), deci=3))
    print('   95% Mean Confidence Interval (t-based) =',
          form.format_interval(stat.get_t_CI(0.05), 3))
    print('   95% Mean Confidence Interval (bootstrap) =',
          form.format_interval(stat.get_bootstrap_CI(0.05, 1000), 3))
    print('   95% Percentile Interval =',
          form.format_interval(stat.get_PI(0.05), 3))


def mytest_summary_stat(data, expected_value, st_dev):
    # define a summary statistics
    sum_stat = stat.SummaryStat(data=data, name='Test summary statistics', )
    print('Testing summary statistics (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(sum_stat)


def mytest_discrete_time(data):
    # define a discrete-time statistics
    discrete_stat = stat.DiscreteTimeStat('Test discrete-time statistics')
    # record data points
    for point in data:
        discrete_stat.record(point)

    print('Testing discrete-time statistics:')
    print_results(discrete_stat)


def mytest_continuous_time(times, observations):
    # define a continuous-time statistics
    continuous_stat = stat.ContinuousTimeStat(initial_time=0, name='Test continuous-time statistics')

    for obs in range(0, len(times)):
        # find the increment
        inc = 0
        if obs == 0:
            inc = observations[obs]
        else:
            inc = observations[obs] - observations[obs - 1]
        continuous_stat.record(times[obs], inc)

    print('Testing continuous-time statistics:')
    print_results(continuous_stat)


def mytest_diff_stat_indp(x, y, expected_value, st_dev):
    # define
    diff_stat = stat.DifferenceStatIndp(x=x, y_ref=y, name='Test DifferenceStatIndp')
    print('Testing DifferenceStatIndp (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(diff_stat)


def mytest_diff_stat_paired(x, y, expected_value, st_dev):
    # define
    diff_stat = stat.DifferenceStatPaired(x=x, y_ref=y, name='Test DifferenceStatPaired')
    print('Testing DifferenceStatPaired (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(diff_stat)


def mytest_ratio_stat_indp(x, y, expected_value, st_dev):
    # define
    ratio_stat = stat.RatioStatIndp(x=x, y_ref=y, name='Test RatioStatIndp')
    print('Testing RatioStatIndp (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(ratio_stat)


def mytest_ratio_stat_paied(x, y, expected_value, st_dev):
    # define
    ratio_stat = stat.RatioStatPaired(x=x, y_ref=y, name='Test RatioStatPaired')

    print('Testing RatioStatPaired (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(ratio_stat)


def mytest_relativeDiff_stat_paied(x, y, expected_value, st_dev):
    # define
    relative_stat = stat.RelativeDifferencePaired(x=x, y_ref=y, name='Test RelativeDifferencePaired')

    print('Testing RelativeDifferencePaired (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(relative_stat)

def mytest_relativeDiff_stat_indp(x, y, expected_value, st_dev):
    # define
    relative_stat = stat.RelativeDifferenceIndp(x=x, y_ref=y, name='Test RelativeDifferenceIndp')

    print('Testing RelativeDifferenceIndp (E = {}, sd = {}):'.format(expected_value, st_dev))
    print_results(relative_stat)

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

# y_ind = numpy.random.normal(5, 2, 1000)
# delta = numpy.random.normal(5, 1, 1000)
# ratio = numpy.random.normal(2, 1, 1000)
# y_ratio_paired = numpy.divide(x, ratio)
# relative_ratio = numpy.random.normal(0.5, 0.1, 1000)
# y_relativeRatio_paired = numpy.divide(x, 1+ratio)


# populate a data set to test continuous-time statistics
sampleT = []
sampleObs = []
i = 0
for i in range(0, 100):
    t = numpy.random.uniform(i, i + 1)
    sampleT.append(t)
    sampleObs.append(10*t)

# test summary statistics
mytest_summary_stat(data=y, expected_value=Y_MEAN, st_dev=Y_SD)

# test discrete-time statistics
mytest_discrete_time(data=y)

# test continuous-time statistics
mytest_continuous_time(times=sampleT, observations=sampleObs)

# test statistics for the difference of two independent samples
mytest_diff_stat_paired(x=x_diff, y=y, expected_value=INCREASE, st_dev=INCREASE_SD)

# test statistics for the ratio of two independent samples
mytest_diff_stat_indp(x=x_diff, y=y,
                      expected_value=INCREASE,
                      st_dev=numpy.sqrt(pow(INCREASE_SD, 2) + 2*pow(Y_SD, 2)))

# test statistics for the difference of two paired samples
mytest_ratio_stat_paied(x=x_ratio, y=y, expected_value=RATIO, st_dev=RATIO_SD)
#
# test statistics for the relative difference of two paired samples
mytest_ratio_stat_indp(x=x_ratio, y=y, expected_value='Unknown', st_dev='Unknown')
#
# test statistics for the ratio of two paired samples
mytest_relativeDiff_stat_paied(x=x_relative_ratio, y=y, expected_value='Unknown', st_dev='Unknown')
#
# test statistics for the relative difference of two independent samples
mytest_relativeDiff_stat_indp(x=x_relative_ratio, y=y, expected_value='Unknown', st_dev='Unknown')
