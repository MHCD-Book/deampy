import numpy as np

from deampy.markov import DiscreteTimeCohortMarkov

# states: 0 = Well, 1 = Mild, 2 = Severe, 3 = Dead
probMatrix = [
    [0.721, 0.202, 0.067, 0.010],
    [0.000, 0.581, 0.407, 0.012],
    [0.000, 0.000, 0.750, 0.250],
    [0.000, 0.000, 0.000, 1.000]
]

# cost and effect parameters (cycle length = 1 year)
entryCost = [0, 500, 2000, 1000]        # one-time cost upon entering each state (e.g., diagnosis, end-of-life care)
stayCost = [100, 1500, 6000, 0]         # annual cost of being in each state
entryQALYsLoss = [0, 0.05, 0.1, 0]      # one-time QALY loss upon entering each state
stayQuality = [1.0, 0.8, 0.5, 0]        # annual QALYs accrued in each state (utility x cycle length)

# create the cohort model
cohort = DiscreteTimeCohortMarkov(
    entry_cost=entryCost,
    stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss,
    stay_quality=stayQuality,
    discount_rate=0.03)

# simulate a cohort of 1000 individuals all starting in the Well state for 20 years
cohort.simulate(
    transition_prob_matrix=probMatrix,
    initial_condition=[1000, 0, 0, 0],
    n_time_steps=20,
    rng=np.random.RandomState(seed=1))

print('Number in each state at the end of simulation:', cohort.get_num_in_states())
print('Total discounted cost:', cohort.get_total_discounted_cost())
print('Total discounted QALYs:', cohort.get_total_discounted_effect())
print('Discounted cost per person:', cohort.get_total_discounted_cost() / 1000)
print('Discounted QALYs per person:', cohort.get_total_discounted_effect() / 1000)


# sanity check: with no transitions, 10 people staying in a state with stay cost 100 and quality 1
# for 3 cycles should accrue 10*100*(1 + 1/1.03 + 1/1.03^2) in cost and 10*(1 + 1/1.03 + 1/1.03^2) in QALYs.
check = DiscreteTimeCohortMarkov(stay_cost=[100, 0], stay_quality=[1, 0], discount_rate=0.03)
check.simulate(transition_prob_matrix=[[1, 0], [0, 1]], initial_condition=[10, 0], n_time_steps=3,
               rng=np.random.RandomState(seed=1))
expectedFactor = sum(pow(1.03, -k) for k in range(3))
assert abs(check.get_total_discounted_cost() - 10 * 100 * expectedFactor) < 1e-9
assert abs(check.get_total_discounted_effect() - 10 * expectedFactor) < 1e-9
print('\nSanity check passed.')
