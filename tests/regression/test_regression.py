import numpy as np

from deampy.regression_models import LinearRegression, RecursiveLinearReg

# testing regression model y = -2 + 3x1 + 1x2
coeffs = [-2, 3, 1]
N = 1000
sigma = 1
l2_reg = 0.02
forgetting_factor = 0.95


def test_reg(X, y, x_to_predict, if_correlated=False):

    test_text = 'y({}) = '.format(x_to_predict)
    y_hat = np.dot(x_to_predict, coeffs)

    # fit a linear regression
    if not if_correlated:
        lr = LinearRegression()
        lr.fit(X=X, y=y)
        print('Regression: ')
        print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
        print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)

    # fit a linear regression with L2 regularization
    lr = LinearRegression(l2_penalty=l2_reg)
    lr.fit(X=X, y=y)
    print('\nRegression (with l2-regularization):')
    print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
    print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)

    # fit a linear regression with forgetting factor
    if not if_correlated:
        lr = LinearRegression()
        lr.fit(X=X, y=y, forgetting_factor=forgetting_factor)
        print('\nRegression (with forgetting factor):')
        print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
        print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)

    # recursive a linear regression
    if not if_correlated:
        lr = RecursiveLinearReg()
        for i in range(N):
            lr.update(x=X[i], y=y[i])
        print('\nRecursive regression: ')
        print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
        print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)

    lr = RecursiveLinearReg(l2_penalty=l2_reg)
    for i in range(N):
        lr.update(x=X[i], y=y[i])
    print('\nRecursive regression (with l2-regularization): ')
    print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
    print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)

    if not if_correlated:
        lr = RecursiveLinearReg()
        for i in range(N):
            lr.update(x=X[i], y=y[i], forgetting_factor=forgetting_factor)
        print('\nRecursive regression (with forgetting factor): ')
        print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
        print(test_text, lr.get_y(x=x_to_predict), 'vs.', y_hat)


if __name__ == "__main__":
    np.random.seed(seed=1)

    # generate X
    X = []
    for n in range(N):
        # each row is a constant 1 and 2 random numbers over [-1, 1]
        x = 2 * np.random.sample(2) - 1
        X.append([1]+list(x))

    # find y's
    y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma, N)

    # test
    test_reg(X, y, x_to_predict= [1, -1, 2])

    # --------------------
    # generate correlated X assuming x1 = 2 * x0
    print('\n\nCorrelated X')
    X = []
    for n in range(N):
        # each row is 3 random numbers over [-1, 1]
        x = 2 * np.random.sample(2) - 1
        x[1] = 2 * x[0]
        X.append([1]+list(x))

    # find y's
    y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma, N)

    # test
    test_reg(X, y, x_to_predict= [1, 2, 4], if_correlated=True)