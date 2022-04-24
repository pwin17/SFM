import numpy as np

def get_essential_matrix(F, K):
    E = np.dot(K.T, np.dot(F,K))
    U, D, Vt = np.linalg.svd(E)
    D = np.diag([1,1,0])
    E = np.dot(U, np.dot(D, Vt))
    return E