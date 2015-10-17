import numpy as np
import scipy.linalg as sl
from mpi4py import MPI
import createMatrix
import plotFunc

'''
This script calculates the temperature in the room using three paralell processes. The temeperature i modelled using the laplacian equation with Dirichlet and Neumann condtions 
'''

def is_outer(rank):
    return rank == 0 or rank == 2
	

def print_solution(solmatrix, sol1, sol2, sol3, sol_rows, sol_cols, n):
	solmatrix[n+1:sol_rows, 0:n] = rearrange_vector(sol1,n,n) 
	solmatrix[0:sol_rows, n:2*n] = rearrange_vector(sol2, 2*n+1, n, 'rotate')
	solmatrix[0:n, 2*n:sol_cols] = rearrange_vector(sol3, n, n, 'rotate') # need to rotat this 
	solmatrix[solmatrix == 0.] = np.nan
	plotFunc.plot_temp(solmatrix)
	
def rearrange_vector(vec, rows, cols, par = "None"):
	if par == 'None':
		return np.fliplr(np.reshape(vec, (rows,cols) ,  order='C'))
	else:
		return np.fliplr(np.reshape(vec, (rows,cols) ,  order='C'))[:,::-1]



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 20
dx = 1/float(n + 1)
omega = 0.8
n_iter = 150

# initializing for first iteration
if is_outer(rank):
    A = createMatrix.generate_outer_matrix(n)
    lu_piv = sl.lu_factor(A)
    sol = np.zeros((n*n, 1))
    bc_send_to_1 = np.zeros((n, 1))
    bc_rec_from_1 = np.zeros((n, 1))
else:
    A = createMatrix.generate_inner_matrix(n)
    lu_piv = sl.lu_factor(A)
    rhs_init = createMatrix.generate_inner_rhs_init(n)
    sol = np.zeros((2*n*n + n, 1))
    bc_send_to_0 = np.zeros((n, 1))
    bc_send_to_2 = np.zeros((n, 1))
    bc_rec_from_0 = np.zeros((n, 1))
    bc_rec_from_2 = np.zeros((n, 1))

for i in range(n_iter):
    if is_outer(rank):
        comm.Recv(bc_rec_from_1, source = 1)
        bc_derivative = (bc_send_to_1 - bc_rec_from_1)/float(2*dx)
        rhs = createMatrix.generate_outer_rhs(n, bc_derivative)
    else:
        rhs = createMatrix.generate_inner_rhs(rhs_init.copy(), bc_rec_from_0, bc_rec_from_2)
    prev_sol = sol
    #sol = np.linalg.solve(A, rhs) # used to be sl.solve(A, rhs)
    sol = sl.lu_solve(lu_piv, rhs) # much faster!
    sol = omega*sol + (1 - omega)*prev_sol
    if is_outer(rank):
        bc_send_to_1 = sol[::n].copy() # copy is needed since mpi requires contigious data to send
        comm.Send(bc_send_to_1, dest = 1)
    else:
        bc_send_to_0 = sol[n*(n+1)::n].copy()
        bc_send_to_2 = sol[n-1:n*n:n].copy()
        comm.Send(bc_send_to_0, dest = 0)
        comm.Send(bc_send_to_2, dest = 2)
        comm.Recv(bc_rec_from_0, source = 0)
        comm.Recv(bc_rec_from_2, source = 2)

if rank == 0:
	comm.Send(sol, dest = 1)

if rank == 2:
	comm.Send(sol, dest = 1)

if rank == 1:
    sol_rows = 2*n + 1
    sol_cols = 3*n
    solmatrix  = np.zeros((2*n +1 ,3*n)) # matrix for the appartment with all inner points
    sol1 = np.zeros((n*n, 1))
    sol3 = np.zeros((n*n, 1))
    sol2 = sol
    comm.Recv(sol1, source = 0)
    comm.Recv(sol3, source = 2)
    #print(sol1, "\n###\n")
    #print(sol2, "\n###\n")
    #print(sol3, "\n###\n")
    print_solution(solmatrix, sol1, sol2, sol3, sol_rows, sol_cols, n)
