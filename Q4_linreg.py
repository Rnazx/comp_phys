from math import log
import matplotlib.pyplot as plt
import math
import numpy as np

f = open('msfit.txt', 'r')
data = np.genfromtxt(f, delimiter='')
f.close()
# print(data)


######################defining functions for chi square fit######################
def chisquare_linear(x, y, sig):
    n = len(sig)
    s, sx, sy, sxx, sxy, syy = 0, 0, 0, 0, 0, 0
    for i in range(n):
        var = (1/sig[i]**2)
        s += (1/sig[i]**2)
        sx += x[i]*var
        sy += y[i]*var
        sxx += (x[i]**2)*var
        syy += (y[i]**2)*var
        sxy += x[i]*y[i]*var
    _del = s*sxx - (sx)**2

    a = ((sxx*sy) - (sx*sxy))/_del
    b = ((s*sxy) - (sx*sy))/_del

    sigmaa = sxx/_del
    sigmab = s/_del

    covarianceb = -sx/_del
    r2 = sxy/(sxx*syy)
    dof = n - 2

    chi2 = 0
    for i in range(n):
        chi2 += (y[i] - a - b*x[i])**2 / sig[i]**2

    return a, b, sigmaa, sigmab, covarianceb, r2, dof, chi2


def plot_func(f, a, b, h):
    X = []
    Y = []
    x = a
    while x < b:
        X.append(x)
        Y.append(f(x))
        x += h
    return X, Y
############################################################################


A = [data[j][1] for j in range(len(data))]
x = [data[j][0] for j in range(len(data))]
sig = [data[j][2] for j in range(len(data))]

e = math.e

log_A = [log(A[i], e) for i in range(len(A))]
sigmaln = [1/sig[i] for i in range(len(sig))]


a, b, sigmaa, sigmab, covarianceb, r2, dof, chi2 = chisquare_linear(
    x, log_A, sigmaln)

print("Slope is (b)", b)
print("y-intercept is (a)", a)

print("Thus the decay constant is negative of the slope ", -b)
lifetime = -1/(b)
print("The mean lifetime of the sample is ", lifetime)
lifetime_err = lifetime**(2) * math.sqrt(sigmab)
print('The error in the lifetime is', lifetime_err)
print('The chi-sqaure value is ', chi2)
print('The value of chi-square critical at 95 lof is ', 1.86,
      'Thus we reject the null hypothesis and the linear fit is not a good fit')


###############Plotting the graph###################################
def f(x):
    return ((b)*x + a)


plt.scatter(x, log_A, label='datapoints')
x, y = plot_func(f, min(x), max(x), 0.001)
plt.plot(x, y, label='chi-square linear-fit', color='orange')
plt.xlabel('time')
plt.ylabel('log(N)')
plt.legend()
plt.show()
