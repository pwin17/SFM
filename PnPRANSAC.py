import numpy as np
from LinearPnP import *
from LinearTriangulation import getP

def reprojection_error(X, x, P1, P2, P3):
    u_reproj = np.divide(np.dot(X, P1), np.dot(X, P3))
    v_reproj = np.divide(np.dot(X, P2), np.dot(X, P3))
    x_reproj = np.hstack((u_reproj, v_reproj))
    error = np.linalg.norm(x - x_reproj)
    return error

def PnPRANSAC(K, x, X, numIter, threshold):
    num_correspondences = 0
    for _ in range(numIter):
        # Randomly select 6 points
        rand_idx = np.random.randint(0, X.shape[0], 6)
        X_hat = X[rand_idx]
        x_hat = x[rand_idx]
        R, C = LinearPnP(K, X_hat, x_hat)
        I = np.eye(3)
        _C = C.reshape(-1,1)
        _C = np.hstack((I, -1*_C))
        P1, P2, P3 = getP(K, R, _C)
        inliers_count = 0
        for j in range(len(X)):
            tempX = np.array([X[j][0],X[j][1],X[j][2],1])
            error = reprojection_error(tempX, x[j], P1, P2, P3)
            if error < threshold:
                inliers_count += 1
        if inliers_count > num_correspondences:
            num_correspondences = inliers_count
            best_C = C
            best_R = R
    return best_R, best_C

