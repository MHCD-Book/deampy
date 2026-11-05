import numpy as np

from deampy.markov import ContinuousTimeCohortMarkov, MultiContinuousTimeCohortMarkov, discrete_to_continuous

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

COHORT_SIZE = 1000
N_SIMS = 100
SEED = 1


# --- 1. fixed transition rate matrix ---
multiCohort = MultiContinuousTimeCohortMarkov(
    transition_rate_matrix=rateMatrix,
    delta_t=1/12,                       # monthly cycles
    entry_cost=entryCost,
    stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss,
    stay_quality=stayQuality,
    discount_rate=0.03)                 # annual discount rate

# simulate 100 replications of a cohort of 1000 individuals in the Well state for 20 years
multiCohort.simulate(
    initial_condition=[COHORT_SIZE, 0, 0, 0],
    n_time_steps=20*12,
    n_sims=N_SIMS,
    seed=SEED)

costs = multiCohort.get_total_discounted_costs()
effects = multiCohort.get_total_discounted_effects()

mean, lower, upper = multiCohort.summarize(costs / COHORT_SIZE)
print('Discounted cost per person: {:.1f} (95% PI: {:.1f}, {:.1f})'.format(mean, lower, upper))
mean, lower, upper = multiCohort.summarize(effects / COHORT_SIZE)
print('Discounted QALYs per person: {:.3f} (95% PI: {:.3f}, {:.3f})'.format(mean, lower, upper))

# number alive over time (times are in years)
times = multiCohort.get_times()
alive = multiCohort.get_sum_size_multiple_states_over_time(state_indices=[0, 1, 2])
mean, lower, upper = multiCohort.summarize(alive)
for k in [0, 60, 120, 240]:
    print('Number alive at year {:.0f}: {:.1f} (95% PI: {:.0f}, {:.0f})'.format(times[k], mean[k], lower[k], upper[k]))

# consistency check: replication i should match a single run with seed (SEED + i)
single = ContinuousTimeCohortMarkov(
    transition_rate_matrix=rateMatrix, delta_t=1/12,
    entry_cost=entryCost, stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss, stay_quality=stayQuality, discount_rate=0.03)
single.simulate(initial_condition=[COHORT_SIZE, 0, 0, 0], n_time_steps=20*12,
                rng=np.random.RandomState(seed=SEED + 3))
assert abs(single.get_total_discounted_cost() - costs[3]) < 1e-6
assert abs(single.get_total_discounted_effect() - effects[3]) < 1e-9
print('\nConsistency check passed.')


# --- 2. dynamic transition rate matrix ---
class TreatmentScaleUp:
    """
    the rate of progression from Mild to Severe declines as treatment is scaled up over time.
    The object keeps a counter of calls (internal state), so it implements reset(),
    which MultiContinuousTimeCohortMarkov calls at the start of each replication.
    """

    def __init__(self, base_rate_matrix):
        self.baseRateMatrix = np.array(base_rate_matrix)
        self.nCalls = 0

    def reset(self):
        self.nCalls = 0

    def get_matrix(self, num_in_states, time):
        self.nCalls += 1
        rates = self.baseRateMatrix.copy()
        # treatment reduces Mild -> Severe by up to 50%, reaching full effect after 5 years
        rates[1, 2] *= 1 - 0.5 * min(time / 5, 1)
        rates[1, 1] = -(rates[1, 0] + rates[1, 2] + rates[1, 3])
        return rates


dynamicMatrix = TreatmentScaleUp(base_rate_matrix=rateMatrix)
multiCohortTreat = MultiContinuousTimeCohortMarkov(
    dynamic_transition_rate_matrix=dynamicMatrix,
    delta_t=1/12,
    entry_cost=entryCost,
    stay_cost=stayCost,
    entry_qalys_loss=entryQALYsLoss,
    stay_quality=stayQuality,
    discount_rate=0.03)

# using the same seed gives common random numbers across the two strategies
multiCohortTreat.simulate(
    initial_condition=[COHORT_SIZE, 0, 0, 0],
    n_time_steps=20*12,
    n_sims=N_SIMS,
    seed=SEED)

assert dynamicMatrix.nCalls == 20*12, 'reset() should be called at the start of each replication.'

# incremental outcomes (treatment scale-up vs. no treatment) in each replication
dCost = (multiCohortTreat.get_total_discounted_costs() - costs) / COHORT_SIZE
dEffect = (multiCohortTreat.get_total_discounted_effects() - effects) / COHORT_SIZE
mean, lower, upper = multiCohortTreat.summarize(dCost)
print('\nIncremental cost per person (scale-up vs. none): {:.1f} (95% PI: {:.1f}, {:.1f})'.format(mean, lower, upper))
mean, lower, upper = multiCohortTreat.summarize(dEffect)
print('Incremental QALYs per person (scale-up vs. none): {:.4f} (95% PI: {:.4f}, {:.4f})'.format(mean, lower, upper))
