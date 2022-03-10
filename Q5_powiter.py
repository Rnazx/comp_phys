


from math import sqrt, pi
import numpy as np
from numpy import cos, sin

f = open('mstrimat.txt', 'r')
A = np.genfromtxt(f, delimiter='')
f.close()


####################################Functions for power method ####################################


def normalize(vec):
    norm = np.linalg.norm(vec, 2)
    return [x / (sqrt(norm)) for x in vec]

def power_method(A, x, tol):
   
    n = len(A)
    x = x / np.linalg.norm(x)
    y = x.copy()

    diff = 1
    while diff > tol:
        xnew = A @ x
        lambda1 = np.dot(xnew, x) / np.dot(x, x)
        xnew = xnew / np.linalg.norm(xnew)
        diff = np.linalg.norm(xnew - x)
        x = xnew.copy()

    vec1 = xnew

    A = A - lambda1 * np.outer(vec1, vec1.T)
    diff = 1
    while diff > tol:
        ynew = A @ y
        lambda2 = np.dot(ynew, y) / np.dot(y, y)
        ynew = ynew / np.linalg.norm(ynew)
        diff = np.linalg.norm(ynew - y)
        y = ynew.copy()

    vec2 = ynew

    return lambda1, lambda2, vec1, vec2


###################################################################################

lambda1, lambda2, vec1, vec2 = power_method(A, np.random.rand(len(A)), 1e-4)

# Given values of eigenvalues and eigenvectors
b = 2
a, c = -1, -1
n = 5

k = 1
given_lambda1 = b + 2 * sqrt(a * c) * cos(k * pi / (n + 1))
given_vec1 = [(2 * (sqrt(c / a)) ** k * sin(k * pi * i / (n + 1))) for i in range(5) ]

k = 2
given_lambda2 = b + 2 * sqrt(a * c) * cos(k * pi / (n + 1))
given_vec2 = [(2 * (sqrt(c / a)) ** k * sin(k * pi * i / (n + 1))) for i in range(5) ]
print('******************************************************************************')
print("First Eigenvalue and eigenvector")
print("Obtained eigenvalue:\t", lambda1)
print("Obtained eigenvector:\t", normalize(vec1))

print()
print("Given eigenvalue:   \t", given_lambda1)
print("Given eigenvector:   \t", normalize(given_vec1))
print('******************************************************************************')
print('******************************************************************************')
print("Second Eigenvalue and eigenvector")
print("Obtained eigenvalue:\t", lambda2)
print("Obtained eigenvector:\t", normalize(vec2))

print()
print("Given eigenvalue:   \t", given_lambda2)
print("Given eigenvector:   \t", normalize(given_vec2))
print('******************************************************************************')
print('There is a descripancy in the eigenvectors corresponding to the second largest eigenvalue. This is because in general, power method is not suitable to find out more than one eigenvector. ')