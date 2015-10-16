# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:30:04 2015

@author: Anton
"""

import numpy as np
import scipy.linalg as sl

def b1_initiate(n, gamma_H = 40, gamma_N = 15, gamma_WF = 5):
    '''Creates an initiate b1 (A1 x = b1) which is to be stored and used in
    b1_generate. A1 refers to the middle room. '''
    
    nelm = 2*n**2 + n    
    dx = 1/(n + 1)

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
            
    return np.divide(b, dx^2)
    
def b_generate(b1_initiated, gamma_1, gamma_2):
    '''Updates b1 with the new Dirichlet conditions. OBS: gamma:s are as in 
    project description.'''
    
    nelm = len(b1_initiated)
    n = len(gamma_1)
    dx = 1/(n + 1)
    first_loop = n**2+2
    
    for k in range(first_loop):
        if k%n == n-1:
            b1_initiated[k] -= gamma_2[k//n]/dx/dx
    
    for k in range(first_loop, nelm):
        if k%n == 0:
            b1_initiated[k] -= gamma_1[(k-1)//n - n]/dx/dx
            
    return b1_initiated

'''
    if k%n == n-1:
            b[k] -= gamma_N'''
            
if __name__ == "__main__":
    b1_initiated = b1_initiate(5)
    print(b1_initiated)
    gamma_1 = np.array([1,2,3,4,5])
    gamma_2 = gamma_1
    new_b = b_generate(b1_initiated, gamma_1, gamma_2) 
    print(new_b)
     
