import numpy as np
import scipy.linalg as sl
from mpi4py import MPI
import createMatrix

def is_outer(rank):
    return rank == 0 or rank == 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("hello this is process", rank)
n = 5
dx = 1/(n + 1)
if is_outer(rank):
    A = createMatrix.generate_outer_matrix(n)
    rhs = createMatrix.generate_outer_rhs(n, np.zeros((n, 1)))
else:
    A = 1
    rhs = 1

n_iter = 10
sol = np.zeros((n*n, 1))
bc_rec = np.zeros((n, 1))
bc_rec_from_0 = np.zeros((n, 1))
bc_rec_from_2 = np.zeros((n, 1))
for i in range(n_iter):
    # calculate solution
    if is_outer(rank):
        bc_derivative = np.zeros((n, 1))
        for j in range(n):
            bc_derivative[j] = (sol[j*n] - bc_rec[j])/2/dx
        if rank == 0:
            bc_derivative *= -1 # one of them have different sign, not sure which
        rhs = createMatrix.generate_outer_rhs(n, bc_derivative)
    else:
        pass
    sol = sl.solve(A, rhs)

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




