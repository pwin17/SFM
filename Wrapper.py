import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

from DataLoader import *
from GetInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EsstentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearPnP import *
from NonlinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *

def main():
    load_data = True
    save_data = True
    data_path = './Data/*.jpg'
    calib_path = './Data/calibration.txt'
    features_path = './Data/matching*.txt'

    # Get camera matrix from calibration file
    K = readK(calib_path)
    print("\nK:\n", K)

    # Get features from matching files
    features_x, features_y, features_matching_map, num_images = loadFeatures(features_path)
    print("\Total features:\n", np.sum(np.sum(features_matching_map)))

    # Get inlier features
    if load_data:
        inlier_features_map = np.load('inlier_features_map.npz')['arr_0']
        with open('F_matrices.pkl', 'rb') as f:
            F_matrices = pickle.load(f)
        
    else:
        # Get inlier features and fundamental matrix for all possible image pairs
        F_matrices, inlier_features_map = getInliersRANSAC(features_x, features_y, features_matching_map, save=save_data)

    print("\Inlier features:\n", np.sum(np.sum(inlier_features_map)))

    # Get fundamental matrix associated with cam pos 1 and 2
    F = F_matrices[(0,1)]
    print("\nF:\n", F)

    # Get essential matrix
    E = get_essential_matrix(F, K)
    print("\nE:\n", E)

    # Get camera pose
    [Rest, Cest] = get_camera_pose(E)
    print("\nC estimate:\n", Cest)
    print("\nR estimate:\n", Rest)

    # Linear Triangulation
    _R = np.eye(3)
    _C = np.zeros((3,1))
    X_all = []
    # Get cam pos 1 and 2 feature pairs
    idx_1_2 = np.array(np.where(inlier_features_map[:,0] & inlier_features_map[:,1])).reshape(-1)

    pts1 = np.hstack((features_x[idx_1_2,0].reshape(-1,1), features_y[idx_1_2,0].reshape(-1,1)))
    pts2 = np.hstack((features_x[idx_1_2,1].reshape(-1,1), features_y[idx_1_2,1].reshape(-1,1)))

    for i in range(len(Cest)):
        Ci = Cest[i]
        Ri = Rest[i]
        X = LinearTriangulation(K, _C, _R, Ci, Ri, pts1, pts2)
        X = X/X[:,3].reshape(-1, 1)
        X_all.append(X)
    print("\nWorld_pts before disambiguation:\n", np.shape(X_all))

    # Disambiguation of camera pose
    R_best, C_best, X_all = DisambiguateCameraPose(X_all, Rest, Cest)
    print("wp best:", np.shape(X_all))
    print("\nR after disambiguation:\n", R_best)
    print("\nC after disambiguation:\n", C_best)

    # Nonlinear Triangulation
    X_3D = NonlinearTriangulation(K, _C, _R, C_best, R_best, pts1, pts2, X_all)
    X_3D = X_3D/X_3D[:,3].reshape(-1, 1)
    print("\nOptimized world_pts:\n", np.shape(X_3D))

    # List of all 3D points found
    X_all = np.zeros((features_x.shape[0],3))

    # Flags of 3D points found in inliers
    flag_3D_points = np.zeros((features_x.shape[0],1),dtype=int).reshape(-1)

    # Setting triangulated 3D points flag to 1, which are inliers found in img1 and img2
    flag_3D_points[idx_1_2] = 1
    X_all[idx_1_2] = X_3D[:,:3]
 
    #Excluding negative Z points
    negative_z_idx = np.where(X_all[:,2]<0)
    flag_3D_points[negative_z_idx] = 0
    print("\n3D points:\n", np.sum(np.sum(flag_3D_points)))

    # Register camera 1 and 2, taking 1 as reference: zero translation and I rotation
    R_set = []
    C_set = []
    C_set.append(_C)
    R_set.append(_R)
    C_set.append(C_best)
    R_set.append(R_best)

    for i in range(2, num_images):
        print(f"----------------------------------PnP for image {i+1}----------------------------------")
        
        found_3D_idx = np.array(np.where(flag_3D_points & inlier_features_map[:,i])).reshape(-1)
        X = X_all[found_3D_idx]
        x = np.hstack((features_x[found_3D_idx,i].reshape(-1,1), features_y[found_3D_idx,i].reshape(-1,1)))
        # print("x shape:",x.shape)
        R_i, C_i = PnPRANSAC(K, x, X, 1000, 5)
        # print("\nR ransac:\n", R_i)
        # print("\nC ransac:\n", C_i)

        R_i, C_i = NonlinearPnp(X, x, K, C_i, R_i)
        # print("R nonlinear:\n", R_i)
        # print("C nonlinear:\n", C_i)
        # Register i-th image/ camera pose:
        C_set.append(C_i)
        R_set.append(R_i)

        for j in range(i):
            # Triangulating all points between i-th image and all previous images sequentially
            idx_j_i = np.array(np.where(inlier_features_map[:,j] & inlier_features_map[:,i])).reshape(-1)
            if len(idx_j_i) < 8:
                continue
            print(f"triangulating img {i+1} and img {j+1} points")
            x_i = np.hstack((features_x[idx_j_i,i].reshape(-1,1), features_y[idx_j_i,i].reshape(-1,1)))
            x_j = np.hstack((features_x[idx_j_i,j].reshape(-1,1), features_y[idx_j_i,j].reshape(-1,1)))
            Ci = C_set[i]
            Ri = R_set[i]
            Cj = C_set[j]
            Rj = R_set[j]

            X_ij = LinearTriangulation(K, Ci, Ri, Cj, Rj, x_i, x_j)
            X_ij = X_ij/X_ij[:,3].reshape(-1, 1)

            X_ij = NonlinearTriangulation(K, Ci, Ri, Cj, Rj, x_i, x_j, X_ij)
            X_ij = X_ij/X_ij[:,3].reshape(-1, 1)

            flag_3D_points[idx_j_i] = 1
            X_all[idx_j_i] = X_ij[:,:3]
            print("appended ", len(idx_j_i), " points between ", j+1 ," and ", i+1)
        
        idx_2D3D, vizM = BuildVisibilityMatrix(flag_3D_points, inlier_features_map, i)
        print("\nVisibility matrix:\n", vizM.shape)
        
        R_set, C_set, X_all = BundleAdjustment(X_all, idx_2D3D, vizM, features_x, features_y, R_set, C_set, K, nCam = i)
        print(f"------------------------------------------------------------------------------------------")
    return
    print("Final Rotations:\n", R_set)
    print("Final Translations:\n", C_set)
    print("\n3D points:\n", np.sum(np.sum(flag_3D_points)))
    print("\nActual 3D points:\n", np.shape(X_all))
    flag_3D_points[X_all[:,2]<0] = 0    
    feature_idx = np.where(flag_3D_points)
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    # For 3D plotting
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    plt.show()
    # plt.savefig(savepath+'3D.png')

main()