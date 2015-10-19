
"""
Mon Oct 19 2015, Project 3 FMNN25

@author: Linus Jangland, Anton Roth and Samuel Wiqvist
"""

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

def plot_solution(sol1, sol2, sol3, n):
    sol_rows = 2*n + 1
    sol_cols = 3*n
    solmatrix  = np.zeros((2*n +1 ,3*n)) # matrix for the appartment with all inner points
    solmatrix[n+1:sol_rows, 0:n] = rearrange_vector(sol1,n,n) 
    solmatrix[0:sol_rows, n:2*n] = rearrange_vector(sol2, 2*n+1, n, 'rotate')
    solmatrix[0:n, 2*n:sol_cols] = rearrange_vector(sol3, n, n, 'rotate') # need to rotat this 
    solmatrix[solmatrix == 0.] = np.nan
    plot_temp(solmatrix)

def rearrange_vector(vec, rows, cols, par = "None"):
    if par == 'None':
        return np.fliplr(np.reshape(vec, (rows,cols) ,  order='C'))
    else:
        return np.fliplr(np.reshape(vec, (rows,cols) ,  order='C'))[:,::-1]

def plot_temp(T):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Temperature')
    plt.imshow(T)
    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    plt.show()


