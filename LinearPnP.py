import numpy as np
'''
Referred to this lecture (Slide 66 and onwards): 
https://www.cis.upenn.edu/~cis580/Spring2015/Lectures/cis580-13-LeastSq-PnP.pdf 
'''
def LinearPnP(K, X, x):
    x = np.hstack((x, np.ones((x.shape[0],1))))
    X = np.hstack((X, np.ones((X.shape[0],1))))

    x_normalized = np.matmul(np.linalg.inv(K), x.T)
    for i in range(len(X)):
        Xi = X[i]

        u, v = x_normalized[0][i], x_normalized[1][i]
        uv_matrix = np.array([[0, -1, v],
                              [1, 0, -1*u],
                              [-1*v, u, 0]])
        Xtilda_matrix = np.array([[*Xi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, *Xi, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *Xi]])
        A = np.matmul(uv_matrix, Xtilda_matrix)
        if i == 0:
            allA = A
        else:
            allA = np.vstack((allA, A))

    allA = np.array(allA, dtype=np.float64)
    _, _, Vt = np.linalg.svd(allA)
    P = Vt[-1].reshape((3, 4))
    
    R = P[:, :3]
    U, _, Vt = np.linalg.svd(R)
    R = np.matmul(U, Vt)

    C = P[:, 3]
    C = np.matmul((-1*R.T), C)

    if np.linalg.det(R) < 0:
        R = -1 * R
        C = -1 * C

    return R, C