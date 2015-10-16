# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:07:43 2015

@author: Anton
"""
import numpy as np
import scipy.linalg as sl
#Big matrix

n = 3 #deltax = 1/(n+1)
elm = 2*n**2 + n #nbr of unknowns

A2 = np.zeros((elm,elm))

A2 = np.diag(-4*np.ones(elm)) + np.diag(np.ones(elm-1),1) + np.diag(np.ones(elm-1),-1)
A2 += np.diag(np.ones(elm-n), n) + np.diag(np.ones(elm-n), -n)

if __name__ == '__main__':
    print(A2)

