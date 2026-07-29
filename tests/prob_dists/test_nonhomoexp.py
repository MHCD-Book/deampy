import numpy as np

from deampy.plots.histogram import plot_histogram
from deampy.random_variates import NonHomogeneousExponential

rng = np.random.RandomState(42)

# non homogeneous exponential random variate generator
nhexp_dist = NonHomogeneousExponential(rates=[0.05, 0.1], time_breaks=[0, 10])

# obtain samples from age 0 (default age)
samples = [nhexp_dist.sample(rng=rng) for i in range(100)]
print(np.average(samples))
plot_histogram(samples)

# obtain samples when the current age is specifeid
samples = [nhexp_dist.sample(rng=rng, arg=50) for _ in range(100)]
print(np.average(samples))
plot_histogram(samples)

# obtain samples from the specified current age over a period
# returns np.nan if the sampled value is beyond the period
samples = [nhexp_dist.sample(rng=rng, arg=(5, 20)) for _ in range(100)]
print(np.nanmean(samples))
plot_histogram(samples)

# obtain samples when the current age is specified and if the sample falls
# beyong the upper bound of the current age group, it return np.nan
samples = [nhexp_dist.sample(rng=rng, arg=(15, None)) for _ in range(100)]
print(np.nanmean(samples))
plot_histogram(samples)