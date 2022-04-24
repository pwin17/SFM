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

def main():
    data_path = './Data/*.jpg'
    calib_path = './Data/calibration.txt'
    features_path = './Data/matching*.txt'

    # Get camera matrix from calibration file
    K = readK(calib_path)
    print("K:", K)

    # Get features from matching files
    mp1, mp2, img_pairs = getFeatures(features_path)
    print("Match points and image pairs(mp1, mp2, pair):", mp1.shape, mp2.shape, img_pairs.shape)

    # Get inlier features
    mp1_inliers, mp2_inliers, img_pairs_inliers = getInliers(mp1, mp2, img_pairs, numIter=100, threshold=0.002)
    print("Inlier match points and image pairs(mp1, mp2, pair):", mp1_inliers.shape, mp2_inliers.shape, img_pairs_inliers.shape)

    # Get feature pairs from first two images to initialize F
    im1im2_feature_pairs = np.where(img_pairs_inliers[:,0] == '1')
    temp_img_pairs_inliers = img_pairs_inliers[im1im2_feature_pairs]
    im1im2_feature_pairs = np.where(temp_img_pairs_inliers[:,1] == '2')
    img1_mp1 = mp1_inliers[im1im2_feature_pairs]
    img2_mp2 = mp2_inliers[im1im2_feature_pairs]
    print("number of matching features between first two images:", np.shape(im1im2_feature_pairs), np.shape(img1_mp1), np.shape(img2_mp2))

    # Get fundamental matrix
    F = estimateFundamentalMatrix(img1_mp1, img2_mp2)
    print("F:", F)

    # Get essential matrix
    E = get_essential_matrix(F, K)
    print("E:", E)

    # Get camera pose
    [Rest, Cest] = get_camera_pose(E)
    print("Cest:", Cest)
    print("Rest:", Rest)

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
    print("world_pts:", np.shape(world_pts))

    # Disambiguation of camera pose
    R_best, C_best = get_best_pose(world_pts, Rest, Cest)
    print("R and C after disambiguation:", R_best, C_best)
    
main()