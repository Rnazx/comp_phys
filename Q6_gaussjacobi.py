import numpy as np
f = open('Q6_A.txt', 'r')
A = np.genfromtxt(f, delimiter='')
f.close()
f = open('Q6_b.txt', 'r')
b = np.genfromtxt(f, delimiter='')
f.close()
# print(A)
# print(b)



#############################Functions to solve the linear equation ################################3
def gauss_siedel(A, b, ep):
    n = len(A)
    x = np.zeros(n)
    x0 = np.ones(n)
    iterations = []
    residue = []
    count = 0  # counts the number of iterations
    while np.linalg.norm(x - x0) > ep:
        iterations.append(count)
        count += 1
        for i in range(n):
            s1, s2 = 0, 0
            for j in range(i):
                s1 += A[i][j] * x[j]
            for j in range(i + 1, n):
                s2 += A[i][j] * x0[j]
            x[i] = 1 / A[i][i] * (b[i] - s1 - s2)
        residue.append(np.linalg.norm(x - x0))
        x0 = x.copy()
    return x


def jacobi(A,b,ep):
    x = []
    for i in range(len(A)):
        x.append(0)
    d = np.diag(A)
    LU = np.array(A) - np.diagflat(d)
    sol = (b - np.dot(LU,x))/d
    while np.linalg.norm(sol - x)>ep: 
        x = sol  
        sol = (b - np.dot(LU,x))/d
    return sol

###############################################################################################3



ep = 1e-5
x_g = gauss_siedel(A,b,ep)
print("The solution obtained using the Gauss Siedel is ",x_g)
x_j = jacobi(A,b,ep)
print("The solution obtained using the Jacobi mathod is ",x_j)
'''The solution obtained using the Gauss Siedel is  [1.12499983768328, -0.49999999993296457, 1.999999999977655, -1.7499999500563939, 1.000000000026814, -0.999999986681705]
The solution obtained using the Jacobi mathod is  [ 1.50000241 -0.5         2.         -2.49999724  1.         -1.00000147]'''