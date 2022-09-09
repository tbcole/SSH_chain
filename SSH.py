import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
import sys

a = 1
t = 1
dt = 0.1*0
N = 10

def Ham(k, t, dt, N, a):
    ham = np.zeros((N, N), dtype=complex)
    ham[0,N-1] = (t-dt)*np.exp(-1j*k*N*a)
    ham[N-1,0] = np.conj(ham[0,N-1])

    for i in range(N):
        for j in range(N):
            if j-i == 1:
                if j%2 == 0:
                    ham[i,j] = t+dt
                else:
                    ham[i,j] = t-dt
                ham[j,i] = np.conj(ham[i,j])
    return ham

num_evals = N
k = np.linspace(-np.pi/(N*a), np.pi/(N*a), 1000)
eigvals = np.zeros((k.shape[0], num_evals))
eigvecs = np.zeros((k.shape[0], N, num_evals), dtype=complex)

for i in range(k.shape[0]):
    H = Ham(k[i], t, dt, N, a)
    eigvals[i,:], eigvecs [i,:,:] = np.linalg.eigh(H)

for i in range(eigvals.shape[1]):
    plt.plot(k, eigvals[:,i])
plt.show()
