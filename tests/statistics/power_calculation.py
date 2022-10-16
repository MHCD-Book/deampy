# Data calculation
# Visualization
import matplotlib.pyplot as plt
import numpy as np
# Power analysis
from statsmodels.stats.power import TTestIndPower

# Initiate the power analysis
power_analysis = TTestIndPower()
# Calculate sample size
sample_size = power_analysis.solve_power(
    effect_size=0.2, alpha=0.05, power=0.8, alternative='two-sided')
# Print results
print('The sample size needed for each group is', round(sample_size))

# Calculate power
power = power_analysis.power(
    effect_size=0.2, alpha=0.05, nobs1=393, ratio=1, alternative='two-sided')
# Print results
print('The power for the hypothesis testing is', round(power, 2))


# Visualization
power_analysis.plot_power(dep_var='nobs',
                          nobs=np.arange(5, 800),
                          effect_size=np.array([0.2, 0.5, 0.8]),
                          alpha=0.05,
                          title='Sample Size vs. Statistical Power')
plt.show()
