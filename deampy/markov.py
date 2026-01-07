import enum

import numpy as np
from numba import jit
from scipy.linalg import logm

from deampy.random_variates import Empirical, Exponential, Multinomial


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
def _continuous_to_discrete_main(trans_rate_matrix, delta_t):

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

    return _continuous_to_discrete_main(trans_rate_matrix=trans_rate_matrix, delta_t=delta_t)


@jit(nopython=True)
def _discrete_to_continuous_main(trans_prob_matrix, delta_t):

    rate_matrix = []
    for i, row in enumerate(trans_prob_matrix):
        rate_row = []  # list of rates
        # calculate rates
        for j in range(len(row)):
            # rate is None for diagonal elements
            if i == j:
                rate = None
            else:
                # rate is zero if this is an absorbing state
                if trans_prob_matrix[i][i] == 1:
                    rate = 0
                else:
                    rate = float(-np.log(trans_prob_matrix[i][i]) * trans_prob_matrix[i][j] / (
                                (1 - trans_prob_matrix[i][i]) * delta_t))
            # append this rate
            rate_row.append(rate)
        # append this row of rates
        rate_matrix.append(rate_row)

    return rate_matrix


def discrete_to_continuous(trans_prob_matrix, delta_t, method='approx'):
    """
    :param trans_prob_matrix: (list of lists) transition probability matrix
    :param delta_t: cycle length
    :param method: method to convert transition probability matrix to transition rate matrix
                   'log': use the matrix logarithm method (default)
                   'approx': use the approximation method
    :return: (list of lists) transition rate matrix
        The approximation method uses:
        Converting [p_ij] to [lambda_ij] where
            lambda_ii = None, and
            lambda_ij = -ln(p_ii) * p_ij / ((1-p_ii)*Delta_t)
    """

    # error checking
    trans_prob_matrix = _assert_prob_matrix(trans_prob_matrix)
    if method == 'approx':
        return _discrete_to_continuous_main(trans_prob_matrix, delta_t)

    elif method == 'log':
        # use the matrix logarithm method

        rate_matrix = logm(trans_prob_matrix)/delta_t
        return rate_matrix

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

    def __init__(self):

        _CohortMarkov.__init__(self)

        self._nonZeroProbs = [] # list of non-zero probabilities
        self._indicesNonZeroProbs = [] # list of state indices with non-zero probabilities

    def initialize(self, initial_condition):

        self._numInStates = initial_condition
        self._numInStatesOverTime = [[] for i in range(len(self._numInStates))]
        self._numToStatesOverTime = [[] for i in range(len(self._numInStates))]

    def condense_prob_matrix(self, transition_prob_matrix):
        """
        Condense the transition probability matrix to include only non-zero probabilities.
        """

        # condense the transition probability matrix to include only non-zero probabilities
        self._nonZeroProbs, self._indicesNonZeroProbs = condense_prob_matrix(
            transition_prob_matrix=transition_prob_matrix)

    def simulate_one_time_step(self, rng=None):

        # store the size of each state
        self.record_number_in_states()

        # initialize the number of transitions to each state
        num_to_states = [0] * len(self._numInStates)
        temp_num_in_states = self._numInStates.copy()

        for s in range(len(self._numInStates)):
            if self._numInStates[s] > 0:
                # find the number of transitions to each state
                binomial = Multinomial(N=self._numInStates[s],
                                       pvals=self._nonZeroProbs[s])
                outs = binomial.sample(rng)
                # update the number of transitions to each state
                for i in range(len(outs)):
                    if self._indicesNonZeroProbs[s][i] != s:
                        num_to_states[self._indicesNonZeroProbs[s][i]] += outs[i]
                # update the number of patients in this state
                temp_num_in_states[s] -= sum(outs)
                # update the number of patients in states
                for i in range(len(outs)):
                    temp_num_in_states[self._indicesNonZeroProbs[s][i]] += outs[i]

        # update the number of patients in each state
        self._numInStates = temp_num_in_states.copy()

        # store the number of transitions to each state
        for i in range(len(self._numInStates)):
            self._numToStatesOverTime[i].append(num_to_states[i])

        self._currentTimeStep += 1

    def record_number_in_states(self):

        # store the size of each state at the end of simulation
        for i in range(len(self._numInStates)):
            self._numInStatesOverTime[i].append(self._numInStates[i])

    def simulate(self, transition_prob_matrix, initial_condition, n_time_steps, rng=None):
        """
        :param transition_prob_matrix: (list of lists) transition probability matrix
        :param initial_condition: initial condition
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param rng: random number generator object
        """

        # error checking
        transition_prob_matrix = _assert_prob_matrix(transition_prob_matrix)

        assert len(initial_condition) == len(transition_prob_matrix), \
            'The length of the initial condition should be equal to the number of states in the transition matrix.'

        # condense the transition probability matrix to include only non-zero probabilities
        self._nonZeroProbs, self._indicesNonZeroProbs = condense_prob_matrix(
            transition_prob_matrix=transition_prob_matrix)

        # initialize the number of patients in each state
        self.initialize(initial_condition=initial_condition)

        for k in range(n_time_steps):
            self.simulate_one_time_step(rng=rng)

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

    def __init__(self, transition_rate_matrix=None, dynamic_transition_rate_matrix=None):
        """
        :param transition_rate_matrix: (list of lists) transition rate matrix
        """

        _CohortMarkov.__init__(self)

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

        self.dtMarkov = DiscreteTimeCohortMarkov()  # discrete-time Markov to evaluate the continuous-time Markov process
        self.deltaT = None

    def simulate(self, initial_condition, delta_t, n_time_steps, rng=None):
        """
        :param initial_condition: (list) initial size of each state
        :param delta_t: (float) cycle length
        :param n_time_steps: (int) number of time steps to simulate the cohort
        :param rng: random number generator object
        """

        self.deltaT = delta_t

        # check if a fixed or a dynamic transition rate matrix is provided
        if self.transRateMatrix is not None:
            transition_prob_matrix = continuous_to_discrete(
                trans_rate_matrix=self.transRateMatrix,
                delta_t=delta_t)

            self.dtMarkov.simulate(
                transition_prob_matrix=transition_prob_matrix,
                initial_condition=initial_condition,
                n_time_steps=n_time_steps,
                rng=rng)

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

                self.dtMarkov.condense_prob_matrix(transition_prob_matrix=transition_prob_matrix)

                # simulate one time step
                self.dtMarkov.simulate_one_time_step(rng=rng)

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