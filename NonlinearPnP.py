import numpy as np
from LinearPnP import *
from scipy.optimize import least_squares
from LinearTriangulation import getP
from scipy.spatial.transform import Rotation

def reprojection_error(x0, K, X, x):
    [qx, qy, qz, qw, c0, c1, c2] = x0
    C = np.array([c0, c1, c2])
    C = np.reshape(C, (3,1))
    I = np.eye(3)
    C = np.hstack((I, -1*C))
    _Rnew = Rotation.from_quat([qx, qy, qz, qw])
    R = _Rnew.as_matrix()
    P1, P2, P3 = getP(K, R, C)

    u_reproj = np.divide(np.dot(X, P1), np.dot(X, P3))
    v_reproj = np.divide(np.dot(X, P2), np.dot(X, P3))
    error = np.square(x[0] - u_reproj) + np.square(x[1] - v_reproj)
    error = np.mean(error.squeeze())
    return error

def NonlinearPnp(X, x, K, C, R):
    X = np.hstack((X, np.ones((X.shape[0],1))))

    _R = Rotation.from_matrix(R)
    qx, qy, qz, qw = _R.as_quat()
    x0 = np.array([qx, qy, qz, qw, C[0], C[1], C[2]])
    optimized_params = least_squares(reprojection_error, x0, args=(K, X, x))
    optimized_WP = optimized_params.x

    [qx, qy, qz, qw, c0, c1, c2] = optimized_WP
    C = np.array([c0, c1, c2])
    _Rnew = Rotation.from_quat([qx, qy, qz, qw])
    R = _Rnew.as_matrix()

    return R, C

