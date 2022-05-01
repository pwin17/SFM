import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from BuildVisibilityMatrix import *
# https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

def getCameraPointIndices(visiblity_matrix):
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

def bundle_adjustment_sparsity(X_index, visiblity_matrix, nCam):
    """
    To create the Sparsity matrix
    """
    number_of_cam = nCam + 1
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    print(m, n)

    i = np.arange(n_observations)
    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = Rotation.from_rotvec(camera_params[i, :3]).as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec

def BundleAdjustment(X_all, X_index, visiblity_matrix, feature_x, feature_y, R_set_, C_set_, K, nCam):
    
    visible3Dpts = X_all[X_index]

    visible_x, visible_y = feature_x[X_index], feature_y[X_index]
    visible2Dpts = []
    for i in range(visiblity_matrix.shape[0]):
        for j in range(visiblity_matrix.shape[1]):
            if visiblity_matrix[i,j] == 1:
                visible2Dpts.append([visible_x[i,j], visible_y[i,j]])
    # print("2d points:", visible2Dpts)
    visible2Dpts = np.array(visible2Dpts).reshape(-1, 2)
    # print('\n 2D Points: \n', visible2Dpts.shape)
    # print('\n 3D Points: \n', visible3Dpts.shape)

    RC_list = []
    for i in range(nCam+1):
        C, R = C_set_[i], R_set_[i]
        _R = Rotation.from_matrix(R)
        ex, ey, ez = _R.as_rotvec()
        RC = [ex, ey, ez, C[0], C[1], C[2]]
        RC_list.append(RC)
    RC_list = np.array(RC_list).reshape(-1, 6)

    x0 = np.hstack((RC_list.ravel(), visible3Dpts.ravel()))
    n_points = visible3Dpts.shape[0]

    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)
    # print('\n Camera Indices: \n', camera_indices.shape)
    # print('\n Camera Indices: \n', camera_indices)
    # print('\n point Indices: \n', point_indices.shape)
    # print('\n point Indices: \n', point_indices)
    
    A = bundle_adjustment_sparsity(X_index, visiblity_matrix, nCam)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_points, camera_indices, point_indices, visible2Dpts, K))

    x1 = res.x
    number_of_cam = nCam + 1
    optimized_camera_params = x1[:number_of_cam * 6].reshape((number_of_cam, 6))
    optimized_points_3d = x1[number_of_cam * 6:].reshape((n_points, 3))

    optimized_X_all = np.zeros_like(X_all)
    optimized_X_all[X_index] = optimized_points_3d

    optimized_C_set, optimized_R_set = [], []
    for i in range(len(optimized_camera_params)):
        _R = Rotation.from_rotvec(optimized_camera_params[i, :3])
        R = _R.as_matrix()
        C = optimized_camera_params[i, 3:].reshape(3,1)
        optimized_C_set.append(C)
        optimized_R_set.append(R)
    
    return optimized_R_set, optimized_C_set, optimized_X_all