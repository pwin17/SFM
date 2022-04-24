import numpy as np
from scipy.optimize import least_squares

def NonlinearTriangulation(K, C0, R0, C1, R1, mp1, mp2, worldpts):
    # Given 2D points, this function computes the 3D point using linear triangulation
    I = np.eye(3)

    C0 = np.hstack((I, -1*C0))
    P0 = np.dot(np.dot(K, R0), C0)

    C1 = C1.reshape(-1, 1)
    C1 = np.hstack((I, -1*C1))
    P1 = np.dot(np.dot(K, R1), C1)

    all_world_points = []
    for i in range(len(worldpts)):
        optimized_params, _ = least_squares(get_reprojection_error, worldpts[i], args=(mp1[i], mp2[i], P0, P1))
        optimized_WP = optimized_params.x
        all_world_points.append(optimized_WP)
    return np.array(all_world_points)

def get_reprojection_error(worldpt, mp1, mp2, P0, P1):
    P01, P02, P03 = P0[0,:].T, P0[1,:].T, P0[2,:].T
    P01, P02, P03 = P01.reshape(-1,1), P02.reshape(-1,1), P03.reshape(-1,1)

    P11, P12, P13 = P1[0,:].T, P1[1,:].T, P1[2,:].T
    P11, P12, P13 = P11.reshape(-1,1), P12.reshape(-1,1), P13.reshape(-1,1)

    # Camera 1
    img1_reprojections = np.divide(np.dot(worldpt, P01), np.dot(worldpt, P03))
    error1 = np.square(mp1[0] - img1_reprojections[0]) + \
        np.square(mp1[1] - img1_reprojections[1])

    # Camera 2
    img2_reprojections = np.divide(np.dot(worldpt, P11), np.dot(worldpt, P13))
    error2 = np.square(mp2[0] - img2_reprojections[0]) + \
        np.square(mp2[1] - img2_reprojections[1])

    total_error = error1 + error2
    return total_error