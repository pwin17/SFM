import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import getP

def NonlinearTriangulation(K, C0, R0, C1, R1, mp1, mp2, worldpts):
    # Given 2D points, this function computes the 3D point using linear triangulation
    I = np.eye(3)
    C0 = C0.reshape(-1,1)
    C0 = np.hstack((I, -1*C0))
    P01, P02, P03 = getP(K, R0, C0)

    C1 = C1.reshape(-1, 1)
    C1 = np.hstack((I, -1*C1))
    P11, P12, P13 = getP(K, R1, C1)

    all_world_points = []
    for i in range(len(worldpts)):
        # print('shapes: ', worldpts[i].shape, mp1[i].shape, mp2[i].shape)
        optimized_params = least_squares(get_reprojection_error, worldpts[i], args=(mp1[i], mp2[i], P01, P02, P03, P11, P12, P13))
        optimized_WP = optimized_params.x
        all_world_points.append(optimized_WP)
    return np.array(all_world_points)

def get_reprojection_error(worldpt, mp1, mp2, P01, P02, P03, P11, P12, P13):

    # Camera 1
    u1_reproj = np.divide(np.dot(worldpt, P01), np.dot(worldpt, P03))
    v1_reproj = np.divide(np.dot(worldpt, P02), np.dot(worldpt, P03))
    error1 = np.square(mp1[0] - u1_reproj) + \
        np.square(mp1[1] - v1_reproj)

    # Camera 2
    u2_reproj = np.divide(np.dot(worldpt, P11), np.dot(worldpt, P13))
    v2_reproj = np.divide(np.dot(worldpt, P12), np.dot(worldpt, P13))
    error2 = np.square(mp2[0] - u2_reproj) + \
        np.square(mp2[1] - v2_reproj)

    total_error = error1 + error2
    return total_error