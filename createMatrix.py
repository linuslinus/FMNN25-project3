import numpy as np
import scipy.linalg as sl

def ind(i, j, n):
# returns the the one dimensional index corresponding to row i, col j
    return (n*i + j) % (n*n)

def generate_outer_matrix(n):
    dx = 1/(n + 1)
    A = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            k = ind(i, j, n)
            A[k][k] += -4
            A[k][ind(i, j - 1, n)] += 1
            A[k][ind(i, j + 1, n)] += 1
            A[k][ind(i - 1, j, n)] += 1
            A[k][ind(i + 1, j, n)] += 1
            if i == 0:
                A[k][ind(i - 1, j, n)] -= 1
            elif i == n - 1:
                A[k][ind(i + 1, j, n)] -= 1
            if j == 0:
                A[k][ind(i, j - 1, n)] -= 1
                A[k][k] += 1 # neumann condition
            elif j == n - 1:
                A[k][ind(i, j + 1, n)] -= 1
    return A/dx/dx
    
def generate_inner_matrix(n): 
	dx = 1/(n + 1) # dirichlet conditions for the large room 
	elm = 2*n**2 + n #nbr of unknowns
	A = np.zeros((elm,elm))
	A = np.diag(-4*np.ones(elm)) + np.diag(np.ones(elm-1),1) + np.diag(np.ones(elm-1),-1)
	A += np.diag(np.ones(elm-n), n) + np.diag(np.ones(elm-n), -n)
	return A/dx/dx

def generate_outer_rhs(n, bc_derivative):
    dx = 1/(n + 1)
    rhs = np.zeros((n*n, 1))
    for i in range(n):
        for j in range(n):
            k = ind(i, j, n)
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
    n = 5
    A = generate_inner_matrix(n)
    print(A)
    #rhs = generate_outer_rhs(n)
    #print(sl.solve(A, rhs))

