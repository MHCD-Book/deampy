import numpy as np

from deampy.plots.histogram import plot_histogram
from deampy.random_variates import NonHomogeneousExponential

rng = np.random.RandomState(42)

# non homogeneous exponential random variate generator
nhexp_dist = NonHomogeneousExponential(rates=[0.05, 0.1], time_breaks=[0, 10])

# obtain samples
samples = [nhexp_dist.sample(rng=rng) for i in range(100)]
print(np.average(samples))
plot_histogram(samples)

# obtain samples
samples = [nhexp_dist.sample(rng=rng, arg=50) for i in range(100)]
print(np.average(samples))
plot_histogram(samples)

# obtain samples
samples = [nhexp_dist.sample(rng=rng, arg=(5, 20)) for i in range(100)]
print(np.nanmean(samples))
plot_histogram(samples)

# obtain samples
samples = [nhexp_dist.sample(rng=rng, arg=(5, None)) for i in range(100)]
print(np.nanmean(samples))
plot_histogram(samples)