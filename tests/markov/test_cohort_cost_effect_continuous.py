import numpy as np

from deampy.markov import ContinuousTimeCohortMarkov, discrete_to_continuous

# states: 0 = Well, 1 = Mild, 2 = Severe, 3 = Dead
# annual transition probability matrix converted to a transition rate matrix (per year)
probMatrix = [
    [0.721, 0.202, 0.067, 0.010],
    [0.000, 0.581, 0.407, 0.012],
    [0.000, 0.000, 0.750, 0.250],
    [0.000, 0.000, 0.000, 1.000]
]
rateMatrix = discrete_to_continuous(trans_prob_matrix=probMatrix, delta_t=1)

# cost and effect parameters (time is measured in years)
entryCost = [0, 500, 2000, 1000]        # one-time cost upon entering each state
stayCost = [100, 1500, 6000, 0]         # annual cost of being in each state
entryQALYsLoss = [0, 0.05, 0.1, 0]      # one-time QALY loss upon entering each state
stayQuality = [1.0, 0.8, 0.5, 0]        # health utility of each state (QALYs per year)

# create the cohort model (monthly cycles)
cohort = ContinuousTimeCohortMarkov(
    transition_rate_matrix=rateMatrix,
    delta_t=1/12,
    entry_cost=entryCost,
    stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss,
    stay_quality=stayQuality,
    discount_rate=0.03)  # annual discount rate

# simulate a cohort of 1000 individuals all starting in the Well state for 20 years with monthly cycles
cohort.simulate(
    initial_condition=[1000, 0, 0, 0],
    n_time_steps=20*12,
    rng=np.random.RandomState(seed=1))

print('Number in each state at the end of simulation:', cohort.get_num_in_states())
print('Total discounted cost:', cohort.get_total_discounted_cost())
print('Total discounted QALYs:', cohort.get_total_discounted_effect())
print('Discounted cost per person:', cohort.get_total_discounted_cost() / 1000)
print('Discounted QALYs per person:', cohort.get_total_discounted_effect() / 1000)


# sanity check: with no transitions, 10 people staying in a state with annual cost 100 and utility 1
# for 2 years with half-year cycles should accrue 10*100*0.5*sum_k 1.03^(-0.5k) in cost (k = 0, 1, 2, 3)
# and 10*0.5*sum_k 1.03^(-0.5k) in QALYs.
check = ContinuousTimeCohortMarkov(
    transition_rate_matrix=[[0, 0], [0, 0]], delta_t=0.5,
    stay_cost=[100, 0], stay_quality=[1, 0], discount_rate=0.03)
check.simulate(initial_condition=[10, 0], n_time_steps=4, rng=np.random.RandomState(seed=1))
expectedFactor = sum(pow(1.03, -0.5 * k) for k in range(4))
assert abs(check.get_total_discounted_cost() - 10 * 100 * 0.5 * expectedFactor) < 1e-9
assert abs(check.get_total_discounted_effect() - 10 * 0.5 * expectedFactor) < 1e-9
print('\nSanity check passed.')
