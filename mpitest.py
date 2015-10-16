import numpy as np
import scipy.linalg as sl
from mpi4py import MPI
import createMatrix
#import plottest

def is_outer(rank):
    return rank == 0 or rank == 2
   
def print_solution(solmatrix, sol1, sol2, sol3, sol_rows, sol_cols, n):
	solmatrix[n+1:sol_rows, 0:n] = rearrage_vector(sol1,n,n) # this part need som extra treatment since the RHS isnt on the correct side in the appartment 
	solmatrix[0:sol_rows, n:2*n] = rearrage_vector(sol2,2*n+1, n)
	solmatrix[0:n, 2*n:sol_cols] = rearrage_vector(sol3,n,n)
	print(solmatrix)
	
def rearrage_vector(vec,rows, cols):
	return np.fliplr(np.reshape(vec, (rows,cols) ,  order='C'))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 5
dx = 1/float(n + 1)
if is_outer(rank):
    A = createMatrix.generate_outer_matrix(n)
    rhs = createMatrix.generate_outer_rhs(n, np.zeros((n, 1)))
    #print("###\n", rhs, "\n###\n")
else:
    A = createMatrix.generate_inner_matrix(n)
    rhs_init = createMatrix.generate_inner_rhs_init(n)
    rhs = rhs_init.copy()
    #print("###\n", A, "\n###\n")

if is_outer(rank):
    sol = np.zeros((n*n, 1))
    prev_sol = sol
    bc_rec = np.zeros((n, 1))
else:
    sol = np.zeros((2*n*n + n, 1))
    prev_sol = sol
    bc_rec_from_0 = np.zeros((n, 1))
    bc_rec_from_2 = np.zeros((n, 1))
n_iter = 10
for i in range(n_iter):
    # calculate solution
    if is_outer(rank):
        bc_derivative = np.zeros((n, 1))
        for j in range(n):
            bc_derivative[j] = (sol[j*n] - bc_rec[j])/float(2*dx)
        rhs = createMatrix.generate_outer_rhs(n, bc_derivative)
    else:
        rhs = createMatrix.generate_inner_rhs(rhs_init.copy(), bc_rec_from_0, bc_rec_from_2)
        #print(rhs, "\n###\n")
    prev_sol = sol
    sol = np.linalg.solve(A, rhs) # used to be sl.solve(A, rhs)
    sol = 0.8*sol + 0.2*prev_sol

    # send data to other processes
    if is_outer(rank):
        bc_send_to_1 = np.zeros((n, 1))
        for j in range(n):
            bc_send_to_1[j] = sol[j*n]
        comm.Send(bc_send_to_1, dest = 1)
        comm.Recv(bc_rec, source = 1)
    else:
        bc_send_to_0 = np.zeros((n, 1))
        bc_send_to_2 = np.zeros((n, 1))
        for j in range(n):
            bc_send_to_0[j] = sol[(n - 1) + j*n]
            bc_send_to_2[j] = sol[n*n + j*n]
        comm.Send(bc_send_to_0, dest = 0)
        comm.Send(bc_send_to_2, dest = 2)
        comm.Recv(bc_rec_from_0, source = 0)
        comm.Recv(bc_rec_from_2, source = 2)

#if rank == 0:
    #print(sol.reshape((n,n)))
    #plottest.plot_temp(sol.reshape((n,n)))
#print("room", rank, ":\n", sol, "\n###\n")

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
	comm.Recv(sol2, source = 2)
	print_solution(solmatrix, sol1, sol2, sol3, sol_rows, sol_cols, n)

