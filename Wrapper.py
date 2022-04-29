import cv2
import numpy as np
import glob

from DataLoader import *
from GetInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EsstentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearPnP import NonlinearPnp
from NonlinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import PnPRANSAC
from NonlinearPnP import NonlinearPnp
from BuildVisibilityMatrix import *

def main():
    data_path = './Data/*.jpg'
    calib_path = './Data/calibration.txt'
    features_path = './Data/matching*.txt'

    # Get camera matrix from calibration file
    K = readK(calib_path)
    print("\nK:\n", K)

    # Get features from matching files
    mp1, mp2, img_pairs, num_images = getFeatures(features_path)
    print("\nMatch points and image pairs(mp1, mp2, pair):\n", mp1.shape, mp2.shape, img_pairs.shape)
    print("mp1:" , mp1[:5])
    print("mp2:" , mp2[:5])
    print("img_pairs:" , img_pairs[:5])

    # Get inlier features
    mp1_inliers, mp2_inliers, img_pairs_inliers = getInliers(mp1, mp2, img_pairs, numIter=100, threshold=0.002)
    print("\nInlier match points and image pairs(mp1, mp2, pair):\n", mp1_inliers.shape, mp2_inliers.shape, img_pairs_inliers.shape)

    # Get feature pairs from first two images to initialize F
    im1im2_feature_pairs = np.where(img_pairs_inliers[:,0] == '1')
    temp_img_pairs_inliers = img_pairs_inliers[im1im2_feature_pairs]
    im1im2_feature_pairs = np.where(temp_img_pairs_inliers[:,1] == '2')
    img1_mp1 = mp1_inliers[im1im2_feature_pairs]
    img2_mp2 = mp2_inliers[im1im2_feature_pairs]
    print("\nNumber of matching features between first two images:\n", np.shape(im1im2_feature_pairs), np.shape(img1_mp1), np.shape(img2_mp2))

    # Save which feature indexes we are using
    feature_pairs_loc = np.zeros((len(mp1_inliers),1), dtype=int)
    feature_pairs_loc[im1im2_feature_pairs] = 1

    # Get fundamental matrix
    F = estimateFundamentalMatrix(img1_mp1, img2_mp2)
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
    world_pts = []
    for i in range(len(Cest)):
        Ci = Cest[i]
        Ri = Rest[i]
        X = LinearTriangulation(K, _C, _R, Ci, Ri, img1_mp1, img2_mp2)
        X = X/X[:,3].reshape(-1, 1)
        world_pts.append(X)
    print("\nWorld_pts:\n", np.shape(world_pts))

    # Disambiguation of camera pose
    R_best, C_best, world_pts_best = DisambiguateCameraPose(world_pts, Rest, Cest)
    print("wp best:", world_pts_best[0])
    print("\nR after disambiguation:\n", R_best)
    print("\nC after disambiguation:\n", C_best)

    # Nonlinear Triangulation
    nlt_world_pts = NonlinearTriangulation(K, _C, _R, C_best, R_best, img1_mp1, img2_mp2, world_pts_best)
    nlt_world_pts = nlt_world_pts/nlt_world_pts[:,3].reshape(-1, 1)
    print("\nOptimized world_pts:\n", np.shape(nlt_world_pts))

    # We have R and C of camera pose 1 and 2 at this point
    C_set = [_C, C_best]
    R_set = [_R, R_best]
    X_all = np.zeros((len(mp1_inliers), X.shape[-1]))
    # feature_pairs_loc[]
    # feature_pairs_loc in line 48 store 0, 1 of unkown and known 3d points
    idx = 0
    for i in range(len(X_all)):
        if feature_pairs_loc[i,0] == 1:
            X_all[i,:] = nlt_world_pts[idx,:]
            idx += 1

    for i in range(2, num_images):
        # Get 2D points associated to the camera i
        img_idx = np.zeros((len(mp1_inliers),1), dtype=int)
        idx0 = np.where(img_pairs_inliers[:,0] == str(i+1))
        idx1 = np.where(img_pairs_inliers[:,1] == str(i+1))
        print(f"total features in Camera {i+1}:", len(idx0[0]) + len(idx1[0]))
        img_idx[idx0] = 1
        img_idx[idx1] = 1
        print(type(img_idx), type(feature_pairs_loc))
        img_idx = np.where(feature_pairs_loc[:,0] & img_idx[:,0])
        # print(f"total features in Camera {i} after filtering:", len(img_idx[0]))
        # feature_pairs_idx_0 = np.bitwise_and(feature_pairs_loc[:,0],img_pairs_inliers[:,0] == str(i+1))
        # feature_pairs_idx_1 = np.bitwise_and(feature_pairs_loc[:,0],img_pairs_inliers[:,1] == str(i+1))
        # feature_pairs_idx_1 = np.where((feature_pairs_loc[:,0]) & (img_pairs_inliers[:,1] == str(i+1)))
        # print(feature_pairs_idx_0)
        print()
        print(img_idx)
        break
        
        img_2d_0 = mp1_inliers[feature_pairs_idx_0]
        feature_pairs_idx_1 = np.where(img_pairs_inliers[:,1] == str(i+1))
        img_2d_1 = mp2_inliers[feature_pairs_idx_1]
        x_i = np.vstack((img_2d_0, img_2d_1))

        idx_all = np.hstack((feature_pairs_idx_0, feature_pairs_idx_1))
        X_i = X_all[idx_all, :]
        print("\nX_i shape:\n", np.shape(X_i))
        print("\nX_i:\n", X_i[:20])

        print("\nx_i shape:\n", np.shape(x_i))
        print("\nx_i:\n", x_i[:20])

        # # PnP RANSAC
        # R_i, C_i = PnPRANSAC(K, x_i, X_i, 100, 5)
        # print("\nR after PnP RANSAC:\n", R_i)
        # print("\nC after PnP RANSAC:\n", C_i)

        # R_i, C_i = NonlinearPnp(X, x_i, K, C_i, R_i)
        # print("\nR after NonlinearPnp:\n", R_i)
        # print("\nC after NonlinearPnp:\n", C_i)
        # break




        
main()