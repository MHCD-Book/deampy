import numpy as np

import deampy.econ_eval as EconEval

np.random.seed(573)

cost_base = np.random.normal(loc=10000, scale=100, size=1000)
effect_base = np.random.normal(loc=2, scale=.1, size=1000)
cost_intervention = np.random.normal(loc=20000, scale=200, size=1000)
effect_intervention = np.random.normal(loc=1, scale=.2, size=1000)

print('')

# ICER calculation assuming paired observations
ICER_paired = EconEval.ICERPaired(cost_intervention, effect_intervention, cost_base, effect_base,
                                  health_measure='d')
print('Paired ICER:\n\tICER: {} '
      '\n\tCI (boostrap): {} '
      '\n\tCI (Bayesian): {} '
      '\n\tCI (Fieller): {} '
      '\n\tCI (Taylor): {} '
      '\n\tPI: '.format(
      ICER_paired.get_ICER(),
      ICER_paired.get_CI(alpha=0.05, num_bootstrap_samples=1000, method='bootstrap'),
      ICER_paired.get_CI(alpha=0.05, num_bootstrap_samples=1000, method='Bayesian'),
      ICER_paired.get_CI(alpha=0.05, method='Fieller'),
      ICER_paired.get_CI(alpha=0.05, method='Taylor'),
      ICER_paired.get_PI(alpha=0.05)))

# ICER calculation assuming independent observations
ICER_indp = EconEval.ICERIndp(cost_intervention, effect_intervention, cost_base, effect_base,
                              health_measure='d')
print('Independent ICER (confidence and prediction interval): \n\t{}\n\t{}\n\t{}'.format(
      ICER_indp.get_ICER(),
      ICER_indp.get_CI(0.05, 1000),
      ICER_indp.get_PI(0.05)))

# try NMB
NMB_paired = EconEval.IncrementalNMBPaired(cost_intervention, effect_intervention, cost_base, effect_base,
                                           health_measure='d')
print('Paired NMB (confidence and prediction interval): \n\t{}\n\t{}\n\t{}'.format(
      NMB_paired.get_incremental_nmb(wtp=10000),
      NMB_paired.get_CI(wtp=10000, alpha=.05),
      NMB_paired.get_PI(wtp=10000, alpha=.05)))

NMB_indp = EconEval.IncrementalNMBIndp(cost_intervention, effect_intervention, cost_base, effect_base,
                                       health_measure='d')
print('Independent NMB (confidence and prediction interval): \n\t{}\n\t{}\n\t{}'.format(
      NMB_indp.get_incremental_nmb(wtp=10000),
      NMB_indp.get_CI(wtp=10000, alpha=.05),
      NMB_indp.get_PI(wtp=10000, alpha=.05)))

print('')
