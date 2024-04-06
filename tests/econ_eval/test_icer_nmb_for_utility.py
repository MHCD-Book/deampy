import numpy as np

import deampy.econ_eval as econ_eval

np.random.seed(573)

# cost and effect observations
cost_base = np.random.normal(loc=10000, scale=1000, size=1000)
effect_base = np.random.normal(loc=1, scale=.1, size=1000)
cost_intervention = np.random.normal(loc=20000, scale=2000, size=1000)
effect_intervention = np.random.normal(loc=2, scale=.2, size=1000)

print('')

# ICER calculation assuming paired observations
icer_paired = econ_eval.ICER_Paired(name='Testing paired ICER',
                                    costs_new=cost_intervention,
                                    effects_new=effect_intervention,
                                    costs_base=cost_base,
                                    effects_base=effect_base)
print('Paired ICER:'
      '\n\tICER: {} '
      '\n\tCI (boostrap): {} '
      '\n\tCI (Bayesian): {} '
      '\n\tPI: {}'.format(
      icer_paired.get_ICER(),
      icer_paired.get_CI(0.05),
      icer_paired.get_CI(0.05, method='Bayesian'),
      icer_paired.get_PI(0.05)))


print('Finding the minimum N:')
print('epsilon', 'min N')
for epsilon in (500, 1000, 2000):
    n = econ_eval.get_min_monte_carlo_param_samples(
        delta_costs=icer_paired._deltaCosts,
        delta_effects=icer_paired._deltaEffects,
        max_wtp=500000,
        epsilon=epsilon,
        alpha=0.05,
        num_bootstrap_samples=1000)
    print(epsilon, n)

# ICER calculation assuming independent observations
ICER_indp = econ_eval.ICER_Indp(costs_new=cost_intervention,
                                effects_new=effect_intervention,
                                costs_base=cost_base,
                                effects_base=effect_base)
print('Independent ICER (confidence and prediction interval): ',
      ICER_indp.get_ICER(),
      ICER_indp.get_CI(0.05),
      ICER_indp.get_PI(0.05, ))

# NMB
NMB_paired = econ_eval.MarginalNMB_Paired(costs_new=cost_intervention,
                                          effects_new=effect_intervention,
                                          costs_base=cost_base,
                                          effects_base=effect_base)
print('Paired NMB (confidence and prediction interval): ',
      NMB_paired.get_marginal_nmb(wtp=10000),
      NMB_paired.get_CI(wtp=10000, alpha=.05),
      NMB_paired.get_PI(wtp=10000, alpha=.05))

NMB_indp = econ_eval.MarginalNMB_Indp(costs_new=cost_intervention,
                                      effects_new=effect_intervention,
                                      costs_base=cost_base,
                                      effects_base=effect_base)
print('Independent NMB (confidence and prediction interval): ',
      NMB_indp.get_marginal_nmb(wtp=10000),
      NMB_indp.get_CI(wtp=10000, alpha=.05),
      NMB_indp.get_PI(wtp=10000, alpha=.05))

print('')
