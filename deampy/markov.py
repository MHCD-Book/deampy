import enum

import numpy as np
from deampy.random_variates import Empirical, Exponential
from deampy.statistics import SummaryStat
from deampy.support.simulation import SeedGenerator
from numba import jit
from scipy.linalg import logm, expm
from scipy.optimize import minimize


@jit(nopython=True)  # nopython=True makes it faster (forces full compilation)
def _out_rate(rates, idx):
    """
    :param rates: list of rates leaving this state
    :param inx: index of this state
    :returns the rate of leaving this state (the sum of rates)
    """

    sum_rates = 0
    for i, v in enumerate(rates):
        if i != idx:
            sum_rates += v
    return sum_rates


@jit(nopython=True)
def condense_prob_matrix(transition_prob_matrix):
    """
    Creates a condensed transition probability matrix where transitions with zero probabilities are removed.
    :param transition_prob_matrix: (list of lists) transition probability matrix
    :return: (tuple) (matrix of nonzero transition probability, matrix of indices with nonzero transition probabilities)
    """

    non_zero_probs = []
    indices_non_zero_probs = []
    for i, probs in enumerate(transition_prob_matrix):
        # find the indices of non-zero probabilities
        row_non_zero_probs = []
        row_non_zero_probs_indices = []
        for j, p in enumerate(probs):
            if p > 0:
                row_non_zero_probs.append(p)
                row_non_zero_probs_indices.append(j)

        non_zero_probs.append(row_non_zero_probs)
        indices_non_zero_probs.append(row_non_zero_probs_indices)

    return non_zero_probs, indices_non_zero_probs


def _assert_prob_matrix(prob_matrix):
    """
    :param prob_matrix: (list of lists or np.ndarray) transition probability matrix
    :return: None if the matrix is valid, otherwise raise an error
    """

    if len(prob_matrix) == 0:
        raise ValueError('An empty probability matrix is provided.')

    assert isinstance(prob_matrix, (list, np.ndarray)), \
        "prob_matrix is a matrix that should be represented as a list of lists or numpy array."

    for i, row in enumerate(prob_matrix):
        assert isinstance(row, (list, np.ndarray)), \
            'prob_matrix should be a list of lists or numpy array. ' \
            'Row {} is not a list or numpy array.'.format(i)

    for i, probs in enumerate(prob_matrix):

        # check if the sum of probabilities in this row is 1.
        s = sum(probs)
        if s < 0.99999 or s > 1.00001:
            raise ValueError('Sum of each row in a probability matrix should be 1. '
                             'Sum of row {} is {}.'.format(i, s))

    if isinstance(prob_matrix, list):
        prob_matrix = np.array(prob_matrix)

    return prob_matrix


def _assert_rate_matrix(rate_matrix):
    """
    :param rate_matrix: (list of lists or np.ndarray) transition rate matrix
    :return: None if the matrix is valid, otherwise raise an error
    """

    if len(rate_matrix) == 0:
        raise ValueError('An empty rate matrix is provided.')

    assert isinstance(rate_matrix, (list, np.ndarray)), \
        "rate_matrix is a matrix that should be represented as a list of lists or numpy array."

    for i, row in enumerate(rate_matrix):
        assert isinstance(row, (list, np.ndarray)), \
            'rate_matrix should be a list of lists or numpy array. ' \
            'Row {} is not a list or numpy array.'.format(i)

    for i, row in enumerate(rate_matrix):
        # make sure all non-diagonal rates are non-negative
        for j, value in enumerate(row):

            if i != j and (value is None or value <0):
                raise ValueError('All non-diagonal rates in a transition rate matrix should be non-negative. '
                                 'Negative rate ({}) found in row index {}.'.format(value, i))
            elif i == j:
                # make sure diagonal rates are None or 0
                if value is None:
                    row[j] = 0.0
                elif value < 0:
                    if abs(sum(row)) > 0.00001:
                        raise ValueError(
                            'The diagonal rate in row {} is negative; therefore the sum of rates in that row should be zero.'.format(i))

    if isinstance(rate_matrix, list):
        rate_matrix = np.array(rate_matrix)

    return rate_matrix


def _assert_dynamic_matrix(rate_matrix):

    # check if rate_matrix has function get_matrix()
    if not hasattr(rate_matrix, 'get_matrix'):
        raise ValueError("The rate_matrix should have a function 'get_matrix' to get the matrix values.")


def get_prob_2_transitions(rates_out, trans_rate_matrix, delta_t):
    """
    returns the the maximum probability of two transitions within delta_t for each state,
        which is used as an upper bound for the error of the approximation method
        in converting a transition rate matrix to a transition probability matrix.
    :param rates_out:
    :param trans_rate_matrix:
    :param delta_t:
    :return:
    """

    # probability that transition occurs within delta_t for each state
    probs_out = []
    for rate in rates_out:
        probs_out.append(1 - np.exp(-delta_t * rate))

    # calculate the probability of two transitions within delta_t for each state
    prob_out_out = []
    for i in range(len(trans_rate_matrix)):

        # probability of leaving state i withing delta_t
        prob_out_i = probs_out[i]

        # probability of leaving the new state after i withing delta_t
        prob_out_again = 0

        for j in range(len(trans_rate_matrix[i])):
            if not i == j:

                # probability of transition from i to j
                prob_i_j = 0
                if rates_out[i] > 0:
                    prob_i_j = trans_rate_matrix[i][j] / rates_out[i]
                # probability of transition from i to j and then out of j within delta_t
                prob_i_j_out = prob_i_j * probs_out[j]
                # update the probability of transition out of i and again leaving the new state
                prob_out_again += prob_i_j_out

        # store the probability of leaving state i to a new state and leaving the new state withing delta_t
        prob_out_out.append(prob_out_i * prob_out_again)

    # return the maximum probability of two transitions within delta_t
    return max(prob_out_out)


@jit(nopython=True)
def __unpack_Q(params, n):
    Q = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] =  params[idx] ** 2 # enforeces q_ij > 0 # np.exp(params[idx]) #
                idx += 1
        Q[i, i] = -np.sum(Q[i, :])  # row sums to 0
    return Q


# @jit(nopython=True)
def __loss(params, p_obs, t):

    q = __unpack_Q(params, n=p_obs.shape[0])
    p_hat = expm(q * t)
    return np.linalg.norm(p_hat - p_obs, ord='fro') ** 2


@jit(nopython=True)
def __continuous_to_discrete_main(trans_rate_matrix, delta_t):

    # list of rates out of each row
    rates_out = []
    for i, row in enumerate(trans_rate_matrix):
        rates_out.append(_out_rate(row, i))

    prob_matrix = np.zeros((len(trans_rate_matrix), len(trans_rate_matrix[0])))
    # prob_matrix = []
    for i in range(len(trans_rate_matrix)):
        # calculate probabilities
        # prob_matrix.append([0] * len(trans_rate_matrix[i]))
        for j in range(len(trans_rate_matrix[i])):
            if i == j:
                prob_matrix[i][j] = float(np.exp(-rates_out[i] * delta_t))
            else:
                if rates_out[i] > 0:
                    prob_matrix[i][j] = float((1 - np.exp(-rates_out[i] * delta_t)) * trans_rate_matrix[i][j] / rates_out[i])

    # # calculate the maximum probability of two transitions within delta_t
    # max_prob_2_transitions = get_prob_2_transitions(
    #     rates_out=rates_out, trans_rate_matrix=trans_rate_matrix, delta_t=delta_t)

    # return the probability matrix and the upper bound for the probability of two transitions with delta_t
    return prob_matrix


def continuous_to_discrete(trans_rate_matrix, delta_t):
    """
    :param trans_rate_matrix: (list of lists) transition rate matrix (assumes None or 0 for diagonal elements)
    :param delta_t: cycle length
    :return: transition probability matrix (list of lists)
             and the upper bound for the probability of two transitions within delta_t (float)
        converting [p_ij] to [lambda_ij] where
            mu_i = sum of rates out of state i
            p_ij = exp(-mu_i*delta_t),      if i = j,
            p_ij = (1-exp(-mu_i*delta_t))*lambda_ij/mu_i,      if i != j.
    """

    # error checking
    trans_rate_matrix = _assert_rate_matrix(trans_rate_matrix)

    return __continuous_to_discrete_main(trans_rate_matrix=trans_rate_matrix, delta_t=delta_t)


@jit(nopython=True)
def __discrete_to_continuous_main(trans_prob_matrix, delta_t):

    # np array filled with 0
    n = len(trans_prob_matrix)
    q = np.zeros((n, n))
    for i, row in enumerate(trans_prob_matrix):
        # calculate rates
        for j in range(len(row)):
            if i != j:
                # rate is zero if this is an absorbing state
                if trans_prob_matrix[i][i] == 1:
                    q[i, j] = 0
                else:
                    q[i, j] = float(-np.log(trans_prob_matrix[i][i]) * trans_prob_matrix[i][j] / (
                                (1 - trans_prob_matrix[i][i]) * delta_t))

    np.fill_diagonal(q, -q.sum(axis=1))

    return q


def discrete_to_continuous(trans_prob_matrix, delta_t, method='approx'):
    """
    :param trans_prob_matrix: (list of lists) transition probability matrix
    :param delta_t: cycle length
    :param method: method to convert transition probability matrix to transition rate matrix
                   'log': use the matrix logarithm method (default)
                   'approx': use the approximation method
                   'optim': use constrained optimization
    :return: (list of lists) transition rate matrix
        The approximation method uses:
        Converting [p_ij] to [lambda_ij] where
            lambda_ii = None, and
            lambda_ij = -ln(p_ii) * p_ij / ((1-p_ii)*Delta_t)
    """

    # error checking
    trans_prob_matrix = _assert_prob_matrix(trans_prob_matrix)
    if method == 'approx':
        return __discrete_to_continuous_main(trans_prob_matrix, delta_t)

    elif method == 'log':
        # use the matrix logarithm method

        rate_matrix = logm(trans_prob_matrix)/delta_t
        return rate_matrix

    elif method == 'optim':

        n = trans_prob_matrix.shape[0]

        # initialize using matrix log
        Q0 = logm(trans_prob_matrix)/delta_t
        # replace negative numbers with 0
        Q0[Q0 < 0] = 0
        # update diagonal elements with the sum of rates out of each state
        np.fill_diagonal(Q0, -Q0.sum(axis=1))
        # list of initial parameters
        initial_params = np.sqrt(Q0[~np.eye(n, dtype=bool)])

        # minimize
        result = minimize(
            __loss,
            initial_params,
            args=(trans_prob_matrix, delta_t),
            method="SLSQP",
        )

        if not result.success:
            raise ValueError("The optimization failed.")

        return __unpack_Q(result.x, len(trans_prob_matrix))

    else:
        raise ValueError("The method should be either 'log' or 'approx'.")


class _Markov:

    def __init__(self, matrix, state_descriptions=None):
        """
        :param state_descriptions: (Enum) description of the states in the format of Enum
        """

        self._n_states = len(matrix)

        if state_descriptions is not None:
            assert type(state_descriptions) is enum.EnumType, 'State description should be an enumeration.'
            assert len(state_descriptions) == len(matrix), \
                ('The number of states in the transition probability/rate matrix '
                 'and the state description should be equal.')

            for i, element in enumerate(state_descriptions):
                assert i == element.value, \
                    'The elements in the state description should be indexed 0, 1, 2, ...' \
                    'The state {} is indexed {} but should be indexed {}.'.format(str(element), element.value, i)

        self._ifStateDescriptionProvided = False if state_descriptions is None else True
        if self._ifStateDescriptionProvided:
            self._states = list(state_descriptions)

    def _error_check(self, current_state_index=None, current_state=None):

        if current_state_index is None and current_state is None:
            raise ValueError('Either current_state_index or current_state should be provided.')

        if current_state_index is not None:
            if not (0 <= current_state_index < self._n_states):
                raise ValueError('The value of the current state index should be greater '
                                 'than 0 and smaller than the number of states. '
                                 'Value provided for current state index is {}.'.format(current_state_index))
        elif current_state is not None:
            if not self._ifStateDescriptionProvided:
                raise ValueError('The state description is not provided; therefore, '
                                 'current_state cannot be used. Please provide current_state_index instead.')
            assert current_state in self._states, \
                'The current state is invalid and not in the state description enumeration.'
            current_state_index = current_state.value

        return current_state_index


class MarkovJumpProcess(_Markov):

    def __init__(self, transition_prob_matrix, state_descriptions=None):
        """
        :param transition_prob_matrix: (list) transition probability matrix of a discrete-time Markov model
        :param state_descriptions: (Enum) description of the states in the format of Enum
        """

        # error checking
        transition_prob_matrix = _assert_prob_matrix(transition_prob_matrix)

        _Markov.__init__(self, matrix=transition_prob_matrix, state_descriptions=state_descriptions)

        # make empirical distributions for each state
        self._empiricalDists = []
        for i, probs in enumerate(transition_prob_matrix):
            # create an empirical distribution over the future states from this state
            self._empiricalDists.append(Empirical(probabilities=probs))

        self._n_states = len(self._empiricalDists)

    def get_next_state(self, current_state_index=None, current_state=None, rng=None):
        """
        :param current_state_index: (int) index of the current state
        :param current_state: (an element of an enumeration) current state
        :param rng: random number generator object
        :return: the index of the next state or the state description of the next state
        """

        # get current state index
        current_state_index = self._error_check(current_state_index=current_state_index, current_state=current_state)

        # find the next state index by drawing a sample from
        # the empirical distribution associated with this state
        next_state_index = self._empiricalDists[current_state_index].sample(rng=rng)

        if self._ifStateDescriptionProvided:
            # if the state description is provided, return the state description
            return self._states[next_state_index]
        else:
            # return the index of the next state
            return next_state_index


class Gillespie(_Markov):

    def __init__(self, transition_rate_matrix, state_descriptions=None):
        """
        :param transition_rate_matrix: transition rate matrix of the continuous-time Markov model
        :param state_descriptions: (Enum) description of the states in the format of Enum
        """

        # error checking
        transition_rate_matrix = _assert_rate_matrix(transition_rate_matrix)

        _Markov.__init__(self, matrix=transition_rate_matrix, state_descriptions=state_descriptions)

        self._rateMatrix = transition_rate_matrix
        self._expDists = []
        self._empiricalDists = []

        for i, row in enumerate(transition_rate_matrix):

            # find sum of rates out of this state
            rate_out = _out_rate(row, i)
            # if the rate is 0, put None as the exponential and empirical distributions
            if rate_out > 0:
                # create an exponential distribution with rate equal to sum of rates out of this state
                self._expDists.append(Exponential(scale=1/rate_out))
                # set the diagonal element to 0 for calculating the probabilities of each event
                row[i] = 0
                # calculate the probability of each event (prob_j = rate_j / (sum over j of rate_j)
                probs = np.array(row) / rate_out
                # create an empirical distribution over the future states from this state
                self._empiricalDists.append(Empirical(probs))

            else:  # if the sum of rates out of this state is 0
                self._expDists.append(None)
                self._empiricalDists.append(None)

    def get_next_state(self, current_state_index=None, current_state=None, rng=None):
        """
        :param current_state_index: index of the current state.
        :param current_state: (an element of an enumeration) current state
        :param rng: random number generator object
        :return: (dt, i) where dt is the time until next event, and i is the index of the next state
                or the state description of the next state.
               It returns None for dt if the process is in an absorbing state
        """

        # get the current state index
        current_state_index = self._error_check(current_state_index=current_state_index, current_state=current_state)

        # if this is an absorbing state (i.e. sum of rates out of this state is 0)
        if self._expDists[current_state_index] is None:
            # the process stays in the current state
            dt = None
            i = current_state_index
        else:
            # find the time until next event
            dt = self._expDists[current_state_index].sample(rng=rng)
            # find the next state
            i = self._empiricalDists[current_state_index].sample(rng=rng)

        i = self._states[i] if self._ifStateDescriptionProvided else i

        return dt, i


class _CohortMarkov:

    def __init__(self):
        self._numInStates = []  # list of state sizes
        self._numInStatesOverTime = []  # list of number of patients in each state over time
        self._numToStatesOverTime = []  # list of number of transitions to each state over time
        self._currentTimeStep = 0

    def get_num_in_states(self):
        """
        :return: list of number of patients in each state
        """
        return self._numInStates


class DiscreteTimeCohortMarkov(_CohortMarkov):

    def __init__(self, entry_cost=None, stay_cost=None, entry_qalys_loss=None, stay_quality=None,
                 discount_rate=0.03):
        """
        :param entry_cost: (list) cost incurred by each individual upon entering each state
        :param stay_cost: (list) cost incurred by each individual for each cycle spent in each state
        :param entry_qalys_loss: (list) QALYs lost by each individual upon entering each state
        :param stay_quality: (list) QALYs accrued by each individual for each cycle spent in each state
                             (i.e., the health utility of each state multiplied by the cycle length)
        :param discount_rate: (float) discount rate per cycle
        Notes on timing:
            - individuals in state s at the beginning of cycle k (k = 0, 1, ...) accrue stay_cost[s] and
              stay_quality[s] for that cycle, discounted by 1/(1+discount_rate)^k.
            - transitions that occur during cycle k are assumed to happen at the end of that cycle,
              so entry_cost and entry_qalys_loss are discounted by 1/(1+discount_rate)^(k+1).
            - no entry cost or QALY loss is incurred for the initial cohort.
        Notes on discount_rate:
            discount_rate should be the rate per cycle. If the cycle is shorter than a year,
            convert the annual discount rate r to the per-cycle rate as:
                per-cycle rate = (1 + r)^(1/n) - 1, where n is the number of cycles in a year.
            For example, for r = 0.03:
                - monthly cycles (n = 12): discount_rate = 1.03^(1/12) - 1 ≈ 0.002466
                - weekly cycles (n = 52): discount_rate = 1.03^(1/52) - 1 ≈ 0.000569
            This ensures that an amount incurred one year from now is discounted by 1/(1+r).
            (Simply dividing r by n, e.g., 0.03/12, slightly overstates the discounting.)
        """

        _CohortMarkov.__init__(self)

        if not (0 <= discount_rate <= 1):
            raise ValueError('discount_rate should be a number between 0 and 1.')

        self._probMatrix = None  # (np.ndarray) transition probability matrix
        self._probMatrixOffDiag = None  # (np.ndarray) transition probability matrix with zero diagonal
        self._nonZeroProbs = []  # (list of np.ndarray) non-zero probabilities of each row
        self._indicesNonZeroProbs = []  # (list of np.ndarray) indices of non-zero probabilities of each row

        # costs and effects are stored as numpy arrays for fast dot products
        self._entryCost = None if entry_cost is None else np.asarray(entry_cost, dtype=float)
        self._stayCost = None if stay_cost is None else np.asarray(stay_cost, dtype=float)
        self._entryQALYsLoss = None if entry_qalys_loss is None else np.asarray(entry_qalys_loss, dtype=float)
        self._stayQuality = None if stay_quality is None else np.asarray(stay_quality, dtype=float)
        self._discountRate = discount_rate

        self.cumCost = 0  # total discounted cost
        self.cumEffect = 0  # total discounted effect (QALYs)

    def initialize(self, initial_condition):

        # check the length of cost and effect parameters
        for name, values in (('entry_cost', self._entryCost),
                             ('stay_cost', self._stayCost),
                             ('entry_qalys_loss', self._entryQALYsLoss),
                             ('stay_quality', self._stayQuality)):
            if values is not None and len(values) != len(initial_condition):
                raise ValueError('The length of {} ({}) should be equal to the number of states ({}).'
                                 .format(name, len(values), len(initial_condition)))

        self._numInStates = np.asarray(initial_condition).copy()
        self._numInStatesOverTime = [[] for i in range(len(self._numInStates))]
        self._numToStatesOverTime = [[] for i in range(len(self._numInStates))]
        self._currentTimeStep = 0

        self.cumCost = 0
        self.cumEffect = 0

    def _collect_cost_and_effect(self, num_in_states, num_to_states):
        """
        updates the total discounted cost and effect for the current cycle
        :param num_in_states: (list) number of individuals in each state at the beginning of this cycle
        :param num_to_states: (list) number of transitions into each state during this cycle
        """

        # discount factors for the beginning and the end of this cycle
        discount_start = pow(1 + self._discountRate, -self._currentTimeStep)
        discount_end = pow(1 + self._discountRate, -(self._currentTimeStep + 1))

        if self._stayCost is not None:
            self.cumCost += discount_start * float(num_in_states @ self._stayCost)
        if self._entryCost is not None:
            self.cumCost += discount_end * float(num_to_states @ self._entryCost)
        if self._stayQuality is not None:
            self.cumEffect += discount_start * float(num_in_states @ self._stayQuality)
        if self._entryQALYsLoss is not None:
            self.cumEffect -= discount_end * float(num_to_states @ self._entryQALYsLoss)

    def get_total_discounted_cost(self):
        """
        :return: total discounted cost of this simulation run
        """
        return self.cumCost

    def get_total_discounted_effect(self):
        """
        :return: total discounted effect (QALYs) of this simulation run
        """
        return self.cumEffect

    def get_num_in_states(self):
        """
        :return: list of number of patients in each state
        """
        return self._numInStates.tolist()

    def set_prob_matrix(self, transition_prob_matrix):
        """
        sets the transition probability matrix used by simulate_one_time_step
        :param transition_prob_matrix: (list of lists or np.ndarray) transition probability matrix
        """

        p = np.array(transition_prob_matrix, dtype=float)
        self._probMatrix = p
        # off-diagonal part (used to count transitions into each state, excluding self-transitions)
        self._probMatrixOffDiag = p.copy()
        np.fill_diagonal(self._probMatrixOffDiag, 0.0)
        # non-zero probabilities of each row (used to sample from multinomial distributions;
        # sampling only over non-zero probabilities keeps the random number stream unchanged)
        self._indicesNonZeroProbs = [np.flatnonzero(row > 0) for row in p]
        self._nonZeroProbs = [row[idx] for row, idx in zip(p, self._indicesNonZeroProbs)]

    def condense_prob_matrix(self, transition_prob_matrix):
        """
        kept for backward compatibility; use set_prob_matrix instead.
        """
        self.set_prob_matrix(transition_prob_matrix=transition_prob_matrix)

    def simulate_one_time_step(self, rng=None, deterministic=False):
        """
        simulates the cohort for one time step (cycle)
        :param rng: random number generator object (ignored if deterministic is True)
        :param deterministic: (bool) if True, the number of transitions out of each state is set to
                              its expected value (state size x transition probability), so state sizes
                              may become non-integer; if False, transitions are sampled from
                              multinomial distributions.
        """

        # store the size of each state
        self.record_number_in_states()

        if deterministic:
            # expected number of transitions (state sizes may become non-integer)
            num_in_states = self._numInStates.astype(float, copy=False)
            # number of transitions to each state (excluding self-transitions)
            num_to_states = num_in_states @ self._probMatrixOffDiag
            # number of patients in each state at the end of this cycle
            new_num_in_states = num_in_states @ self._probMatrix
        else:
            num_in_states = self._numInStates
            # counts[s, j] = number of transitions from state s to state j during this cycle
            counts = np.zeros(self._probMatrix.shape, dtype=np.int64)
            for s in np.flatnonzero(num_in_states > 0):
                counts[s, self._indicesNonZeroProbs[s]] = rng.multinomial(
                    num_in_states[s], self._nonZeroProbs[s])
            new_num_in_states = counts.sum(axis=0)
            num_to_states = new_num_in_states - np.diag(counts)

        # collect the discounted cost and effect of this cycle
        self._collect_cost_and_effect(num_in_states=num_in_states, num_to_states=num_to_states)

        # update the number of patients in each state
        self._numInStates = new_num_in_states

        # store the number of transitions to each state
        for i, n in enumerate(num_to_states.tolist()):
            self._numToStatesOverTime[i].append(n)

        self._currentTimeStep += 1

    def record_number_in_states(self):

        # store the size of each state
        for i, n in enumerate(self._numInStates.tolist()):
            self._numInStatesOverTime[i].append(n)

    def simulate(self, transition_prob_matrix, initial_condition, n_time_steps, rng=None, deterministic=False):
        """
        :param transition_prob_matrix: (list of lists) transition probability matrix
        :param initial_condition: initial condition
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param rng: random number generator object (ignored if deterministic is True)
        :param deterministic: (bool) if True, simulates the deterministic (expected-value) cohort model;
                              if False, simulates the stochastic model (see simulate_one_time_step)
        """

        # error checking
        transition_prob_matrix = _assert_prob_matrix(transition_prob_matrix)

        assert len(initial_condition) == len(transition_prob_matrix), \
            'The length of the initial condition should be equal to the number of states in the transition matrix.'

        self.set_prob_matrix(transition_prob_matrix=transition_prob_matrix)

        # initialize the number of patients in each state
        self.initialize(initial_condition=initial_condition)

        for k in range(n_time_steps):
            self.simulate_one_time_step(rng=rng, deterministic=deterministic)

        # store the size of each state at the end of simulation
        self.record_number_in_states()

    def get_times(self):
        """
        :return: list of time steps where the size of each state is stored
        """
        return range(self._currentTimeStep + 1)

    def get_periods(self):
        """
        :return: list of time periods where the numbers entering each state is stored
        """
        return range(1, self._currentTimeStep+1)

    def get_state_size_over_time(self, state_index):
        """
        :return: list of state sizes
        """
        return self._numInStatesOverTime[state_index]

    def get_transition_to_states_over_time(self, state_index):
        """
        :return: list of number of transitions to each state
        """

        return self._numToStatesOverTime[state_index]

    def get_transitions_to_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: sum of the transitions to the states in the list
        """

        sum_transitions = np.zeros(self._currentTimeStep)
        for i in state_indices:
            sum_transitions += np.array(self._numToStatesOverTime[i])
        return  sum_transitions

    def get_sum_size_multiple_states(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: sum of the sizes of the states in the list
        """

        sum_size = 0
        for i in state_indices:
            sum_size += self._numInStates[i]

        return sum_size

    def get_sum_size_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: sum of the sizes of the states in the list
        """

        sum_size = np.zeros(self._currentTimeStep + 1)
        for i in state_indices:
            sum_size += np.array(self._numInStatesOverTime[i])

        return sum_size


class ContinuousTimeCohortMarkov(_CohortMarkov):

    def __init__(self, transition_rate_matrix=None, dynamic_transition_rate_matrix=None, delta_t=None,
                 entry_cost=None, stay_cost=None, entry_qalys_loss=None, stay_quality=None,
                 discount_rate=0.03):
        """
        :param transition_rate_matrix: (list of lists) transition rate matrix
        :param dynamic_transition_rate_matrix: object with function get_matrix(num_in_states, time)
                                               that returns the transition rate matrix
        :param delta_t: (float) cycle length used to simulate the continuous-time Markov process
        :param entry_cost: (list) cost incurred by each individual upon entering each state
        :param stay_cost: (list) cost incurred by each individual per unit of time spent in each state
                          (e.g., annual cost if time is measured in years)
        :param entry_qalys_loss: (list) QALYs lost by each individual upon entering each state
        :param stay_quality: (list) health utility of each state
                             (QALYs accrued per unit of time spent in each state)
        :param discount_rate: (float) discount rate per unit of time (e.g., annual discount rate)
        Notes:
            with cycle length delta_t, stay_cost and stay_quality are multiplied by delta_t
            and the per-cycle discount rate is (1 + discount_rate)^delta_t - 1, so that an amount
            incurred at time t is discounted by 1/(1+discount_rate)^t.
            See DiscreteTimeCohortMarkov for the timing assumptions within each cycle.
        """

        _CohortMarkov.__init__(self)

        if delta_t is None or delta_t <= 0:
            raise ValueError('delta_t should be a positive number.')
        if not (0 <= discount_rate <= 1):
            raise ValueError('discount_rate should be a number between 0 and 1.')

        if transition_rate_matrix is None and dynamic_transition_rate_matrix is None:
            raise ValueError('Either transition_rate_matrix or dynamic_transition_rate_matrix should be provided.')

        if transition_rate_matrix is not None and dynamic_transition_rate_matrix is not None:
            raise ValueError('Only one of transition_rate_matrix or dynamic_transition_rate_matrix should be provided.')

        # error checking
        if transition_rate_matrix is not None:
            transition_rate_matrix = _assert_rate_matrix(transition_rate_matrix)
        if dynamic_transition_rate_matrix is not None:
            _assert_dynamic_matrix(dynamic_transition_rate_matrix)

        self.transRateMatrix = transition_rate_matrix
        self.dynamicTransRateMatrix = dynamic_transition_rate_matrix

        self.deltaT = delta_t

        # discrete-time Markov to evaluate the continuous-time Markov process
        # (with per-cycle costs, QALYs, and discount rate)
        self.dtMarkov = DiscreteTimeCohortMarkov(
            entry_cost=entry_cost,
            stay_cost=None if stay_cost is None else [c * delta_t for c in stay_cost],
            entry_qalys_loss=entry_qalys_loss,
            stay_quality=None if stay_quality is None else [q * delta_t for q in stay_quality],
            discount_rate=pow(1 + discount_rate, delta_t) - 1)

    def simulate(self, initial_condition, n_time_steps, rng=None, deterministic=False):
        """
        :param initial_condition: (list) initial size of each state
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param rng: random number generator object (ignored if deterministic is True)
        :param deterministic: (bool) if True, the number of transitions out of each state in each cycle
                              is set to its expected value (so state sizes may become non-integer);
                              if False, transitions are sampled from multinomial distributions.
        """

        delta_t = self.deltaT

        # check if a fixed or a dynamic transition rate matrix is provided
        if self.transRateMatrix is not None:
            transition_prob_matrix = continuous_to_discrete(
                trans_rate_matrix=self.transRateMatrix,
                delta_t=delta_t)

            self.dtMarkov.simulate(
                transition_prob_matrix=transition_prob_matrix,
                initial_condition=initial_condition,
                n_time_steps=n_time_steps,
                rng=rng,
                deterministic=deterministic)

        # if a dynamic transition rate matrix is provided
        else:
            self.dtMarkov.initialize(initial_condition=initial_condition)
            k = 0
            while k < n_time_steps:
                # get the transition rate matrix at this time step
                rate_matrix = self.dynamicTransRateMatrix.get_matrix(
                    num_in_states=self.dtMarkov.get_num_in_states(), time=k*delta_t)

                transition_prob_matrix = continuous_to_discrete(
                    trans_rate_matrix=rate_matrix,
                    delta_t=delta_t)

                self.dtMarkov.set_prob_matrix(transition_prob_matrix=transition_prob_matrix)

                # simulate one time step
                self.dtMarkov.simulate_one_time_step(rng=rng, deterministic=deterministic)

                k += 1

            # store the size of each state at the end of simulation
            self.dtMarkov.record_number_in_states()

    def get_times(self):
        """
        :return: list of time points where the size of each state is stored
        """
        return [i*self.deltaT for i in self.dtMarkov.get_times()]

    def get_periods(self):
        """
        :return: list of time periods where the numbers entering each state is stored
        """
        return self.dtMarkov.get_periods()

    def get_state_size_over_time(self, state_index):
        """
        :return: list of state sizes
        """
        return self.dtMarkov.get_state_size_over_time(state_index)

    def get_transition_to_states_over_time(self, state_index):
        """
        :return: list of number of transitions to each state
        """
        return self.dtMarkov.get_transition_to_states_over_time(state_index)

    def get_transitions_to_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: sum of the transitions to the states in the list
        """
        return self.dtMarkov.get_transitions_to_multiple_states_over_time(state_indices)

    def get_sum_size_multiple_states(self, state_indices):
        """
        :param state_indices: list of state indices
        :return: sum of people in the specified states
        """
        return self.dtMarkov.get_sum_size_multiple_states(state_indices)

    def get_sum_size_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of state indices
        :return: sum of people in the specified states
        """
        return self.dtMarkov.get_sum_size_multiple_states_over_time(state_indices)

    def get_num_in_states(self):
        """
        :return: list of number of patients in each state
        """
        return self.dtMarkov.get_num_in_states()

    def get_total_discounted_cost(self):
        """
        :return: total discounted cost of this simulation run
        """
        return self.dtMarkov.get_total_discounted_cost()

    def get_total_discounted_effect(self):
        """
        :return: total discounted effect (QALYs) of this simulation run
        """
        return self.dtMarkov.get_total_discounted_effect()


class _MultiCohortMarkov:
    """
    base class to simulate a cohort Markov model multiple times and record the outcomes of each replication.
    Outcomes are stored as numpy arrays where the first dimension is the replication:
        costs: (n_sims,) total discounted cost of each replication
        effects: (n_sims,) total discounted effect (QALYs) of each replication
        numInStatesOverTime: (n_sims, n_states, n_time_steps + 1) size of each state over time
        numToStatesOverTime: (n_sims, n_states, n_time_steps) number of transitions to each state over time
    """

    def __init__(self, cohort_model):
        """
        :param cohort_model: (DiscreteTimeCohortMarkov or ContinuousTimeCohortMarkov) the model to
                             simulate in each replication (re-initialized at the start of each replication)
        """

        self._model = cohort_model
        self.costs = None
        self.effects = None
        self.numInStatesOverTime = None
        self.numToStatesOverTime = None
        self._times = None
        self._periods = None

    def _simulate_replications(self, simulate_one, n_states, n_time_steps, n_sims, seed):
        """
        :param simulate_one: function that receives a random number generator and simulates one replication
        :param n_states: (int) number of states
        :param n_time_steps: (int) number of time steps
        :param n_sims: (int) number of replications
        :param seed: (int or None) if provided, it is used to generate the seeds of replications
                     (using SeedGenerator), which makes the results reproducible and allows
                     common random numbers when comparing strategies
        """

        if not (isinstance(n_sims, (int, np.integer)) and n_sims > 0):
            raise ValueError('n_sims should be a positive integer.')

        self.costs = np.zeros(n_sims)
        self.effects = np.zeros(n_sims)
        self.numInStatesOverTime = np.zeros((n_sims, n_states, n_time_steps + 1))
        self.numToStatesOverTime = np.zeros((n_sims, n_states, n_time_steps))

        # generate the seed of each replication
        seeds = SeedGenerator().next_seeds(n=n_sims, rng=np.random.RandomState(seed=seed))

        for i in range(n_sims):

            rng = np.random.RandomState(seed=seeds[i])

            # simulate one replication
            simulate_one(rng)

            # record the outcomes of this replication
            self.costs[i] = self._model.get_total_discounted_cost()
            self.effects[i] = self._model.get_total_discounted_effect()
            for s in range(n_states):
                self.numInStatesOverTime[i, s, :] = self._model.get_state_size_over_time(s)
                self.numToStatesOverTime[i, s, :] = self._model.get_transition_to_states_over_time(s)

        self._times = list(self._model.get_times())
        self._periods = list(self._model.get_periods())

    def _check_simulated(self):
        if self.costs is None:
            raise RuntimeError('The model has not been simulated yet. Call simulate() first.')

    @staticmethod
    def summarize(values, alpha=0.05):
        """
        :param values: (np.ndarray) outcomes where the first dimension is the replication
        :param alpha: (float) significance level for the confidence and percentile intervals
        :return: (mean, confidence interval, percentile interval) across replications, calculated
                 using deampy.statistics.SummaryStat, where the confidence interval is the
                 t-based (1-alpha) confidence interval of the mean and the percentile interval
                 is the (1-alpha) percentile interval.
                 If values is 1-dimensional, mean is a float and each interval is a list [l, u].
                 Otherwise, mean is an np.ndarray with the shape of values[0] and each interval
                 is an np.ndarray with the shape of values[0] + (2,), so that interval[..., 0]
                 and interval[..., 1] are the lower and upper bounds.
        """
        values = np.asarray(values, dtype=float)

        if values.ndim == 1:
            stat = SummaryStat(data=values)
            return stat.get_mean(), stat.get_t_CI(alpha=alpha), stat.get_PI(alpha=alpha)

        # summarize each outcome (e.g., each time point) across replications
        out_shape = values.shape[1:]
        flat = values.reshape(values.shape[0], -1)
        means = np.zeros(flat.shape[1])
        conf_intervals = np.zeros((flat.shape[1], 2))
        percentile_intervals = np.zeros((flat.shape[1], 2))
        for j in range(flat.shape[1]):
            stat = SummaryStat(data=flat[:, j])
            means[j] = stat.get_mean()
            conf_intervals[j] = stat.get_t_CI(alpha=alpha)
            percentile_intervals[j] = stat.get_PI(alpha=alpha)

        return (means.reshape(out_shape),
                conf_intervals.reshape(out_shape + (2,)),
                percentile_intervals.reshape(out_shape + (2,)))

    def get_n_sims(self):
        self._check_simulated()
        return len(self.costs)

    def get_times(self):
        """
        :return: list of time points where the size of each state is stored
        """
        self._check_simulated()
        return self._times

    def get_periods(self):
        """
        :return: list of time periods where the numbers entering each state is stored
        """
        self._check_simulated()
        return self._periods

    def get_total_discounted_costs(self):
        """
        :return: (np.ndarray) total discounted cost of each replication
        """
        self._check_simulated()
        return self.costs

    def get_total_discounted_effects(self):
        """
        :return: (np.ndarray) total discounted effect (QALYs) of each replication
        """
        self._check_simulated()
        return self.effects

    def get_num_in_states(self):
        """
        :return: (np.ndarray of shape (n_sims, n_states)) number in each state at the end of each replication
        """
        self._check_simulated()
        return self.numInStatesOverTime[:, :, -1]

    def get_state_size_over_time(self, state_index):
        """
        :return: (np.ndarray of shape (n_sims, n_time_steps + 1)) size of the state over time in each replication
        """
        self._check_simulated()
        return self.numInStatesOverTime[:, state_index, :]

    def get_transition_to_states_over_time(self, state_index):
        """
        :return: (np.ndarray of shape (n_sims, n_time_steps)) number of transitions to the state
                 over time in each replication
        """
        self._check_simulated()
        return self.numToStatesOverTime[:, state_index, :]

    def get_sum_size_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: (np.ndarray of shape (n_sims, n_time_steps + 1)) sum of the sizes of the states
                 over time in each replication
        """
        self._check_simulated()
        return self.numInStatesOverTime[:, list(state_indices), :].sum(axis=1)

    def get_transitions_to_multiple_states_over_time(self, state_indices):
        """
        :param state_indices: list of indices of states
        :return: (np.ndarray of shape (n_sims, n_time_steps)) sum of the transitions to the states
                 over time in each replication
        """
        self._check_simulated()
        return self.numToStatesOverTime[:, list(state_indices), :].sum(axis=1)


class MultiDiscreteTimeCohortMarkov(_MultiCohortMarkov):

    def __init__(self, entry_cost=None, stay_cost=None, entry_qalys_loss=None, stay_quality=None,
                 discount_rate=0.03):
        """
        simulates a DiscreteTimeCohortMarkov model multiple times
        (see DiscreteTimeCohortMarkov for the description of parameters)
        """

        _MultiCohortMarkov.__init__(
            self, cohort_model=DiscreteTimeCohortMarkov(
                entry_cost=entry_cost, stay_cost=stay_cost,
                entry_qalys_loss=entry_qalys_loss, stay_quality=stay_quality,
                discount_rate=discount_rate))

    def simulate(self, transition_prob_matrix, initial_condition, n_time_steps, n_sims, seed=None):
        """
        :param transition_prob_matrix: (list of lists) transition probability matrix
        :param initial_condition: (list) initial size of each state
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param n_sims: (int) number of replications
        :param seed: (int or None) used to generate the seeds of replications (using SeedGenerator)
        """

        # error checking once (instead of in every replication)
        transition_prob_matrix = _assert_prob_matrix(transition_prob_matrix)

        self._simulate_replications(
            simulate_one=lambda rng: self._model.simulate(
                transition_prob_matrix=transition_prob_matrix,
                initial_condition=initial_condition,
                n_time_steps=n_time_steps,
                rng=rng),
            n_states=len(initial_condition),
            n_time_steps=n_time_steps,
            n_sims=n_sims,
            seed=seed)


class MultiContinuousTimeCohortMarkov(_MultiCohortMarkov):

    def __init__(self, transition_rate_matrix=None, dynamic_transition_rate_matrix=None, delta_t=None,
                 entry_cost=None, stay_cost=None, entry_qalys_loss=None, stay_quality=None,
                 discount_rate=0.03):
        """
        simulates a ContinuousTimeCohortMarkov model multiple times
        (see ContinuousTimeCohortMarkov for the description of parameters)
        Note: if dynamic_transition_rate_matrix keeps an internal state that changes during a simulation run,
              it should implement a method reset(), which is called at the start of each replication.
        """

        _MultiCohortMarkov.__init__(
            self, cohort_model=ContinuousTimeCohortMarkov(
                transition_rate_matrix=transition_rate_matrix,
                dynamic_transition_rate_matrix=dynamic_transition_rate_matrix,
                delta_t=delta_t,
                entry_cost=entry_cost, stay_cost=stay_cost,
                entry_qalys_loss=entry_qalys_loss, stay_quality=stay_quality,
                discount_rate=discount_rate))

    def simulate(self, initial_condition, n_time_steps, n_sims, seed=None):
        """
        :param initial_condition: (list) initial size of each state
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param n_sims: (int) number of replications
        :param seed: (int or None) used to generate the seeds of replications (using SeedGenerator)
        """

        dynamic_matrix = self._model.dynamicTransRateMatrix

        def simulate_one(rng):
            if dynamic_matrix is not None and hasattr(dynamic_matrix, 'reset'):
                dynamic_matrix.reset()
            self._model.simulate(initial_condition=initial_condition, n_time_steps=n_time_steps, rng=rng)

        self._simulate_replications(
            simulate_one=simulate_one,
            n_states=len(initial_condition),
            n_time_steps=n_time_steps,
            n_sims=n_sims,
            seed=seed)