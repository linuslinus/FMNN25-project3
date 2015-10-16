import numpy as np
import scipy.linalg as sl

n = 4
# deltax = 1/(n + 1)

def ind(i, j):
# returns the the one dimensional index corresponding to row i, col j
    return (n*i + j) % (n*n)

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

if __name__ == '__main__':
    print(A)


