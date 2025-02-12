from math import pow, exp

import matplotlib.pyplot as plt

from deampy.plots.plot_support import output_figure


class _StepSizeRule:

    def get_epsilon(self, itr):
        # goes to zero as itr goes to infinity
        pass

    def get_learning_rate(self, itr):
        # goes to 1 as itr goes to infinity

        return 1 - self.get_epsilon(itr)


class EpsilonGreedy(_StepSizeRule):

    def __init__(self, formula, beta, min=0, max=1):
        """
            For selecting the greedy action with probability 1-epsilon.
            for pow decay formula: epsilon_n = min + (max-min)/n^beta, beta over (0.5, 1], n > 0
            for exp decay formula: epsilon_n = min + (max-min) * exp(-beta * n), beta > 0, n > 0
        :param formula: (str) 'pow' or 'exp'
        :param beta: (float) the decay rate
        :param min: (float) the minimum epsilon
        :param max: (float) the maximum epsilon
        """

        if formula not in ['pow', 'exp']:
            raise ValueError('Invalid formula. Choose between "pow" and "exp".')
        if beta <= 0:
            raise ValueError('Beta should be greater than 0.')
        if min < 0 or max > 1:
            raise ValueError('Epsilon should be between 0 and 1.')

        self._formula = formula
        self._beta = beta
        self._min = min
        self._max = max

    def __str__(self):
        """
        :return: (str) the formula and the parameters
        """
        return '{}-beta{}-max{}-min{}'.format(self._formula, self._beta, self._max, self._min)

    def get_epsilon(self, itr):

        if itr <= 0:
            return self._max

        if self._formula == 'pow':
            return self._min + (self._max - self._min) * pow(itr, -self._beta)
        elif self._formula == 'exp':
            return self._min + (self._max - self._min) * exp(-self._beta * itr)
        else:
            raise ValueError('Invalid formula. Choose between "pow" and "exp".')

    @staticmethod
    def plot(formula, betas, maxs, mins, n_itrs, fig_size=None, fig_filename=None):
        """
        plots the exploration and learning rates
        :param formula: (str) 'pow' or 'exp'
        :param betas: (list) of decay rates
        :param maxs: (list) of maximum epsilon values
        :param mins: (list) of minimum epsilon values
        :param n_itrs: (int) number of iterations
        :param fig_size: (tuple) the size of the figure
        :param fig_filename: (str) the filename to save the figure
        """

        x = range(1, n_itrs + 1)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)

        for max in maxs:
            for min in mins:
                for beta in betas:
                    rule = EpsilonGreedy(formula=formula, beta=beta, max=max, min=min)
                    y = [rule.get_epsilon(i) for i in x]
                    ax[0].plot(x, y, label=str(rule))
                    y = [rule.get_learning_rate(i) for i in x]
                    ax[1].plot(x, y, label=str(rule))

        ax[0].axhline(y=0, color='black', linestyle='--')
        ax[1].axhline(y=1, color='black', linestyle='--')

        ax[0].set_xlabel('Iteration')
        ax[1].set_xlabel('Iteration')

        ax[0].set_ylabel('Exploration Rate')
        ax[1].set_ylabel('Learning Rate')

        # ax.set_title('Epsilon-Greedy Exploration Rule')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        if fig_filename:
            output_figure(fig, fig_filename)
        else:
            fig.show()


class Harmonic(_StepSizeRule):
    # step_n = b / (b + n - 1), for n > 0 and b >= 1
    # (i is the iteration of the optimization algorithm)

    def __init__(self, b):
        self._b = b

    def __str__(self):
        return 'b{}'.format(self._b)

    def get_epsilon(self, itr):
        if itr <= 0:
            return 1
        return self._b / (self._b + itr - 1)

    @staticmethod
    def plot(bs, n_itrs):

        x = range(1, n_itrs + 1)

        fig, ax = plt.subplots()
        for b in bs:
            rule = Harmonic(b)
            y = [rule.get_learning_rate(i) for i in x]
            ax.plot(x, y, label=str(rule))
        ax.plot(x, y)
        ax.axhline(y=1, color='black', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Forgetting Factor')
        ax.set_title('Harmonic Learning Rule')
        ax.legend()
        fig.show()


if __name__ == '__main__':

    EpsilonGreedy.plot(
        formula='exp', maxs=[0.5], mins=[0.025], betas=[0.01, 0.02, .03], n_itrs=1000)
    EpsilonGreedy.plot(
        formula='pow', maxs=[0.5], mins=[0.025], betas=[0.5, 0.7, 0.9], n_itrs=1000)

    # Harmonic.plot(bs=[1, 10, 20], n_itrs=1000)
