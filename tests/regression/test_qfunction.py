import numpy as np

from deampy.regression_models import PolynomialQFunction

N = 1000
sigma = 0.05 # standard deviation of the noise
l2_reg = 0.01
forgetting_factor = 1
np.random.seed(seed=1)


# testing regression model
# y = 2 + x1 - x2 + x1^2 + 4x1x2 - 8x2^2
# and x1 and x2 are continuous variables
coeffs = [2, 1, -1, 1, 4, -8] # coefficients of this regression model

# generate X
x1s = []
x2s = []
X = [] # for the regression
for n in range(N):
    # get 2 random numbers over [-1, 1] for x1 and x2
    x = 2*np.random.sample(2)-1
    # store observations on x1, and x2
    x1s.append(x[0])
    x2s.append(x[1])

    # store the row for the regression
    X.append([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])

# find y's
y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma, N)

# fit a polynomial regression
q = PolynomialQFunction(degree=2, l2_penalty=l2_reg)
for i in range(N):
    q.update(y=y[i], x=[x1s[i], x2s[i]])

print('Regression with only continuous variables: ')
print('True Coeffs:', coeffs, 'vs.\nEsti Coeffs: ', q.get_coeffs())
print('True y([2, -1]) = ', np.dot([1, 2, -1, 4, -2, 1], coeffs), 'vs.\n'
      'Esti y([2, -1]) = ', q.f([2, -1]))


# --------------------
# testing regression model
# y = 2 + x1 - x2 + x1^2 + 4x1x2 - 8x2^2
#  + I(-1, 2x1 - 3x2 + x1^2 - 2x1x2 + 6x2^2)
# and x1 and x2 are continuous variables and I is a binary variable
# coefficients of this regression model
coeffs = [2, 1, -1, 1, 4, -8,
          -1, 2, -3, 1, -2, 6]

# generate X
Is = []
x1s = []
x2s = []
X = [] # for the regression
for n in range(N):
    # get a random 0 or 1 for I
    I = np.random.randint(low=0, high=2)
    # get 2 random numbers over [-1, 1] for x1 and x2
    x = 2*np.random.sample(2)-1
    # store observations on I, x1, and x2
    Is.append(I)
    x1s.append(x[0])
    x2s.append(x[1])

    # store the row for the regression
    X.append([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2,
              I, I*x[0], I*x[1], I*x[0]**2, I*x[0]*x[1], I*x[1]**2])

# find y's
y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma, N)

# fit a polynomial regression
q = PolynomialQFunction(degree=2, l2_penalty=l2_reg)
for i in range(N):
    q.update(y=y[i],
             x=[x1s[i], x2s[i]],
             binaries=[Is[i]])

print('\nRegression with both: ')
print('True Coeffs:', coeffs, 'vs.\nEsti Coeffs: ', q.get_coeffs())






#
# # testing regression model
# # y = 2 + 5I + x1^2 + 4x1x2 - 8x2^2
# # where I is a binary variable that takes 0 or 1 values
# # and x1 and x2 are continuous variables
# coeffs = [2, 5, 1, 4, -8] # coefficients of this regression model
#
# # generate X
# Is = []
# x1s = []
# x2s = []
# X = [] # for the regression
# for n in range(N):
#
#     # get a random 0 or 1 for I
#     I = np.random.randint(low=0, high=1)
#     # get 2 random numbers over [-1, 1] for x1 and x2
#     x = 2*np.random.sample(2)-1
#
#     # store observations on I, x1, and x2
#     Is.append(I)
#     x1s.append(x[0])
#     x2s.append(x[1])
#
#     # store the row for the regression
#     X.append([1, I, x[0]**2, x[0]*x[1], x[1]**2])
#
# # find y's
# y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma, N)
#
# # fit a polynomial regression
# q = PolynomialQFunction(degree=2, l2_penalty=l2_reg)
# for i in range(N):
#     q.update(y=y[i],
#              values_of_continuous_features=[x1s[i], x2s[i]],
#              values_of_indicator_features=[Is[i]])
#
# print('Regression: ')
# print('Coeffs: ', q.get_coeffs(), 'vs.', coeffs)
