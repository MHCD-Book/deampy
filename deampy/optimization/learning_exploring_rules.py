from math import pow, exp

import matplotlib.pyplot as plt


class _ExplorationRule:

    def get_epsilon(self, itr):
        pass


class _LearningRule:

    def get_step_size(self, itr):
        # goes to zero as itr goes to infinity
        return 0

    def get_forgetting_factor(self, itr):
        # goes to 1 as itr goes to infinity
        return 1/(1+self.get_step_size(itr))


class EpsilonGreedy(_ExplorationRule):
    # For selecting the greedy action with probability 1-epsilon.
    # for pow decay formula: epsilon_n = min + (max-min)/n^beta, beta over (0.5, 1], n > 0
    # for exp decay formula: epsilon_n = min + (max-min) * exp(-beta * n), beta > 0, n > 0

    def __init__(self, formula, beta, min=0, max=1):
        """
        :param formula: (str) 'pow' or 'exp'
        :param beta: (float) the decay rate
        :param min: (float) the minimum epsilon
        :param max: (float) the maximum epsilon
        """
        self._formula = formula
        self._beta = beta
        self._min = min
        self._max = max

    def __str__(self):
        return '{}-beta{}-max{}-min{}'.format(self._formula, self._beta, self._max, self._min)

    def get_epsilon(self, itr):

        if self._formula == 'pow':
            return self._min + (self._max - self._min) * pow(itr, -self._beta)
        elif self._formula == 'exp':
            return self._min + (self._max - self._min) * exp(-self._beta * itr)
        else:
            raise ValueError('Invalid formula. Choose between "power" and "exponential".')

    @staticmethod
    def plot(formula, betas, maxes, mins, n_itrs):

        x = range(1, n_itrs + 1)

        fig, ax = plt.subplots()

        for max in maxes:
            for min in mins:
                for beta in betas:
                    rule = EpsilonGreedy(formula=formula, beta=beta, max=max, min=min)
                    y = [rule.get_epsilon(i) for i in x]
                    ax.plot(x, y, label=str(rule))
        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Epsilon')
        ax.set_title('Epsilon-Greedy Exploration Rule')
        ax.legend()
        fig.show()


class Harmonic(_LearningRule):
    # step_n = b / (b + n), for n >= 0 and b >= 1
    # (i is the iteration of the optimization algorithm)

    def __init__(self, b):
        self._b = b

    def __str__(self):
        return 'b{}'.format(self._b)

    def get_step_size(self, itr):
        return self._b / (self._b + itr - 1)

    @staticmethod
    def plot(bs, n_itrs):

        x = range(1, n_itrs + 1)

        fig, ax = plt.subplots()
        for b in bs:
            rule = Harmonic(b)
            y = [rule.get_forgetting_factor(i) for i in x]
            ax.plot(x, y, label=str(rule))
        ax.plot(x, y)
        ax.axhline(y=1, color='black', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Forgetting Factor')
        ax.set_title('Harmonic Learning Rule')
        ax.legend()
        fig.show()


if __name__ == '__main__':

    EpsilonGreedy.plot(formula='exp', maxes=[1], mins=[0.05], betas=[0.01, 0.02, .03], n_itrs=1000)
    EpsilonGreedy.plot(formula='pow', maxes=[1], mins=[0.05], betas=[0.5, 0.7, 0.9], n_itrs=1000)

    Harmonic.plot(bs=[1, 10, 20], n_itrs=1000)
