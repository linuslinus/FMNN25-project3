import numpy as np
import scipy.linalg as sl

def ind(i, j):
# returns the the one dimensional index corresponding to row i, col j
    return (n*i + j) % (n*n)

def generate_outer_matrix(n):
    dx = 1/(n + 1)
    A = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            k = ind(i, j)
            A[k][k] += -4
            A[k][ind(i, j - 1)] += 1
            A[k][ind(i, j + 1)] += 1
            A[k][ind(i - 1, j)] += 1
            A[k][ind(i + 1, j)] += 1
            if i == 0:
                A[k][ind(i - 1, j)] -= 1
            elif i == n - 1:
                A[k][ind(i + 1, j)] -= 1
            if j == 0:
                A[k][ind(i, j - 1)] -= 1
                A[k][k] += 1 # neumann condition
            elif j == n - 1:
                A[k][ind(i, j + 1)] -= 1
    return A/dx/dx

def generate_outer_rhs(n):
    dx = 1/(n + 1)
    bc_derivative = np.ones((n, 1))
    rhs = np.zeros((n*n, 1))
    for i in range(n):
        for j in range(n):
            k = ind(i, j)
            if i == 0:
                rhs[k] += -15
            elif i == n - 1:
                rhs[k] += -15
            if j == 0:
                rhs[k] += dx*bc_derivative[i]
            elif j == n - 1:
                rhs[k] += -40
    return rhs/dx/dx

if __name__ == '__main__':
    n = 4
    A = generate_outer_matrix(n)
    rhs = generate_outer_rhs(n)
    print(sl.solve(A, rhs))
