
"""
Mon Oct 19 2015, Project 3 FMNN25

@author: Linus Jangland, Anton Roth and Samuel Wiqvist
"""

import numpy as np

def ind(i, j, n):
# returns the the one dimensional index corresponding to row i, col j
    return (n*i + j) % (n*n)

def generate_outer_matrix(n):
    dx = 1/float(n + 1)
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
    
def generate_outer_rhs(n, bc_derivative):
    dx = 1/float(n + 1)
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
    
def generate_inner_matrix(n): 
    dx = 1/float(n + 1) # dirichlet conditions for the large room 
    nelm = 2*n**2 + n #nbr of unknowns
    A = np.diag(-4*np.ones(nelm)) + np.diag(np.ones(nelm-n), n) + np.diag(np.ones(nelm-n), -n)
    sup_sub = np.ones(nelm - 1)
    for k in range(1, 2*n + 1):
        sup_sub[n*k -1] = 0
    A += np.diag(sup_sub, 1) + np.diag(sup_sub, -1)
    return A/dx/dx

def generate_inner_rhs_init(n, gamma_H = 40, gamma_N = 15, gamma_WF = 5):
    '''Creates an initiate b1 (A1 x = b1) which is to be stored and used in
    b1_generate. A1 refers to the middle room. '''
    
    nelm = 2*n**2 + n
    dx = 1/float(n + 1)
    
    b = np.zeros((nelm, 1))
    first_loop = n**2+2
    
    for k in range(first_loop):
        if k//n == 0:
            b[k] -= gamma_H
        if k%n == 0:
            b[k] -= gamma_N
    
    for k in range(first_loop, nelm):
        if k//n == 2*n:
            b[k] -= gamma_WF
        if k%n == n-1:
            b[k] -= gamma_N
            
    return b/dx/dx
    
def generate_inner_rhs(inner_rhs_initiated, gamma_1, gamma_2):
    '''Updates b1 with the new Dirichlet conditions. OBS: gamma:s are as in 
    project description.'''
    
    nelm = len(inner_rhs_initiated)
    n = len(gamma_1)
    dx = 1/float((n + 1))
    first_loop = n**2+2
    
    for k in range(first_loop):
        if k%n == n-1:
            inner_rhs_initiated[k] -= gamma_2[k//n]/(dx*dx) #k//n
    
    for k in range(first_loop, nelm):
        if k%n == 0:
            inner_rhs_initiated[k] -= gamma_1[(k-1)//n - n]/dx/dx # (k-1)//n - n
            
    return inner_rhs_initiated
