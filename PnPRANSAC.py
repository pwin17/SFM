import numpy as np
from LinearPnP import *
from LinearTriangulation import getP

def reprojection_error(X, x, P1, P2, P3):
    u1_reproj = np.divide(np.dot(X, P1), np.dot(X, P3))
    v1_reproj = np.divide(np.dot(X, P2), np.dot(X, P3))
    error = np.square(x[0] - u1_reproj) + np.square(x[1] - v1_reproj)
    return error

def PnPRANSAC(K, x, X, numIter, threshold):
    num_correspondences = 0
    for _ in range(numIter):
        # Randomly select 8 points
        rand_idx = np.random.randint(0, X.shape[0], 6)
        X_hat = X[rand_idx]
        x_hat = x[rand_idx]
        R, C = LinearPnP(K, X_hat, x_hat)
        I = np.eye(3)
        C = C.reshape(-1,1)
        C = np.hstack((I, -1*C))
        P1, P2, P3 = getP(K, R, C)
        inliers_count = 0
        for j in range(len(X)):
            error = reprojection_error(X[j], x[j], P1, P2, P3)
            if error < threshold:
                inliers_count += 1
        if inliers_count > num_correspondences:
            num_correspondences = inliers_count
            best_C = C
            best_R = R
    return best_R, best_C[:,-1]

