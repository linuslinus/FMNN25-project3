import numpy as np
import scipy.linalg as sl
from mpi4py import MPI
import createMatrix
import plotting

'''
This script calculates the temperature in the room using three paralell processes. The temeperature i modelled using the laplacian equation with Dirichlet and Neumann condtions 
'''

def is_outer(rank):
# returns true if rank corresponds to one of the outer rooms
    return rank == 0 or rank == 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 30
dx = 1/float(n + 1)
omega = 0.8
n_iter = 10

# initializing for first iteration
if is_outer(rank):
    A = createMatrix.generate_outer_matrix(n)
    sol = np.zeros((n*n, 1))
    bc_rec_from_1 = np.zeros((n, 1))
else:
    A = createMatrix.generate_inner_matrix(n)
    rhs_init = createMatrix.generate_inner_rhs_init(n)
    sol = np.zeros((2*n*n + n, 1))
    bc_rec_from_0 = np.zeros((n, 1))
    bc_rec_from_2 = np.zeros((n, 1))
lu_piv = sl.lu_factor(A)

for i in range(n_iter):
    if is_outer(rank):
        comm.Recv(bc_rec_from_1, source = 1)
        rhs = createMatrix.generate_outer_rhs(n, bc_rec_from_1)
    else:
        rhs = createMatrix.generate_inner_rhs(rhs_init.copy(), bc_rec_from_0, bc_rec_from_2)
    prev_sol = sol
    #sol = np.linalg.solve(A, rhs)
    sol = sl.lu_solve(lu_piv, rhs)
    sol = omega*sol + (1 - omega)*prev_sol
    if is_outer(rank):
        bc_send_to_1 = sol[::n].copy() # copy is needed since mpi requires contigious data to send
        comm.Send(bc_send_to_1, dest = 1)
    else:
        bc_send_to_0 = (sol[n*(n+1)::n] - sol[n*(n+1)+1::n])/dx
        bc_send_to_2 = (sol[n-1:n*n:n] - sol[n-2:n*n:n])/dx
        comm.Send(bc_send_to_0, dest = 0)
        comm.Send(bc_send_to_2, dest = 2)
        comm.Recv(bc_rec_from_0, source = 0)
        comm.Recv(bc_rec_from_2, source = 2)

if is_outer(rank):
    comm.Send(sol, dest = 1)
else:
    sol1 = np.zeros((n*n, 1))
    sol2 = sol
    sol3 = np.zeros((n*n, 1))
    comm.Recv(sol1, source = 0)
    comm.Recv(sol3, source = 2)
    plotting.plot_solution(sol1, sol2, sol3, n)
