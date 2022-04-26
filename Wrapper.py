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
from NonlinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import PnPRANSAC

def main():
    data_path = './Data/*.jpg'
    calib_path = './Data/calibration.txt'
    features_path = './Data/matching*.txt'

    # Get camera matrix from calibration file
    K = readK(calib_path)
    print("\nK:\n", K)

    # Get features from matching files
    mp1, mp2, img_pairs = getFeatures(features_path)
    print("\nMatch points and image pairs(mp1, mp2, pair):\n", mp1.shape, mp2.shape, img_pairs.shape)

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
    R_best, C_best, world_pts_best = get_best_pose(world_pts, Rest, Cest)
    print("wp best:", world_pts_best[0])
    print("\nR after disambiguation:\n", R_best)
    print("\nC after disambiguation:\n", C_best)

    # Nonlinear Triangulation
    _R = np.eye(3)
    _C = np.zeros((3,1))
    nlt_world_pts = NonlinearTriangulation(K, _C, _R, C_best, R_best, img1_mp1, img2_mp2, world_pts_best)
    nlt_world_pts = nlt_world_pts/nlt_world_pts[:,3].reshape(-1, 1)
    print("\nOptimized world_pts:\n", np.shape(nlt_world_pts))

    # PnP RANSAC
    R_best, C_best = PnPRANSAC(K, img1_mp1, nlt_world_pts, 100, 5)
    print("\nR after PnP RANSAC:\n", R_best)
    print("\nC after PnP RANSAC:\n", C_best)
    
main()