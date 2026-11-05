import numpy as np

from deampy.markov import DiscreteTimeCohortMarkov, MultiDiscreteTimeCohortMarkov

# states: 0 = Well, 1 = Mild, 2 = Severe, 3 = Dead
probMatrix = [
    [0.721, 0.202, 0.067, 0.010],
    [0.000, 0.581, 0.407, 0.012],
    [0.000, 0.000, 0.750, 0.250],
    [0.000, 0.000, 0.000, 1.000]
]

# cost and effect parameters (cycle length = 1 year)
entryCost = [0, 500, 2000, 1000]        # one-time cost upon entering each state
stayCost = [100, 1500, 6000, 0]         # annual cost of being in each state
entryQALYsLoss = [0, 0.05, 0.1, 0]      # one-time QALY loss upon entering each state
stayQuality = [1.0, 0.8, 0.5, 0]        # annual QALYs accrued in each state (utility x cycle length)

COHORT_SIZE = 1000
N_SIMS = 200
SEED = 1

# create the multi-cohort model
multiCohort = MultiDiscreteTimeCohortMarkov(
    entry_cost=entryCost,
    stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss,
    stay_quality=stayQuality,
    discount_rate=0.03)

# simulate 200 replications of a cohort of 1000 individuals all starting in the Well state for 20 years
multiCohort.simulate(
    transition_prob_matrix=probMatrix,
    initial_condition=[COHORT_SIZE, 0, 0, 0],
    n_time_steps=20,
    n_sims=N_SIMS,
    seed=SEED)

# outcomes of each replication
costs = multiCohort.get_total_discounted_costs()        # shape (n_sims,)
effects = multiCohort.get_total_discounted_effects()    # shape (n_sims,)
print('Number of replications:', multiCohort.get_n_sims())
print('Discounted cost per person in the first 5 replications:', costs[:5] / COHORT_SIZE)

# mean and 95% percentile interval across replications
mean, lower, upper = multiCohort.summarize(costs / COHORT_SIZE, alpha=0.05)
print('Discounted cost per person: {:.1f} (95% PI: {:.1f}, {:.1f})'.format(mean, lower, upper))
mean, lower, upper = multiCohort.summarize(effects / COHORT_SIZE, alpha=0.05)
print('Discounted QALYs per person: {:.3f} (95% PI: {:.3f}, {:.3f})'.format(mean, lower, upper))

# state sizes over time across replications
dead = multiCohort.get_state_size_over_time(state_index=3)      # shape (n_sims, n_time_steps + 1)
mean, lower, upper = multiCohort.summarize(dead)
print('\nNumber dead at year 10: {:.1f} (95% PI: {:.0f}, {:.0f})'.format(mean[10], lower[10], upper[10]))

# number alive (Well + Mild + Severe) over time
alive = multiCohort.get_sum_size_multiple_states_over_time(state_indices=[0, 1, 2])
mean, lower, upper = multiCohort.summarize(alive)
for t in [0, 5, 10, 20]:
    print('Number alive at year {}: {:.1f} (95% PI: {:.0f}, {:.0f})'.format(t, mean[t], lower[t], upper[t]))

# incident severe cases over time
newSevere = multiCohort.get_transition_to_states_over_time(state_index=2)   # shape (n_sims, n_time_steps)
print('Mean number of new severe cases in year 1:', multiCohort.summarize(newSevere)[0][0])


# consistency check: replication i should match a single run with seed (SEED + i)
for i in [0, 7]:
    single = DiscreteTimeCohortMarkov(
        entry_cost=entryCost, stay_cost=stayCost,
        entry_qalys_loss=entryQALYsLoss, stay_quality=stayQuality, discount_rate=0.03)
    single.simulate(transition_prob_matrix=probMatrix, initial_condition=[COHORT_SIZE, 0, 0, 0],
                    n_time_steps=20, rng=np.random.RandomState(seed=SEED + i))
    assert abs(single.get_total_discounted_cost() - costs[i]) < 1e-6
    assert abs(single.get_total_discounted_effect() - effects[i]) < 1e-9
    assert np.array_equal(single.get_state_size_over_time(3), dead[i])
print('\nConsistency check passed.')
