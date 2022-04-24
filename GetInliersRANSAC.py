import numpy as np
from EstimateFundamentalMatrix import *

def getInliers(mp1, mp2, img_pairs, numIter, threshold):
    mp1_inlier = []
    mp2_inlier = []
    new_img_pairs = []
    inliers_count = 0

    for _ in range(numIter):
        # Randomly select 8 points
        rand_idx = np.random.randint(0, mp1.shape[0], 8)
        rand_mp1 = np.array([mp1[rand_idx[0]], mp1[rand_idx[1]], mp1[rand_idx[2]], mp1[rand_idx[3]], mp1[rand_idx[4]], mp1[rand_idx[5]], mp1[rand_idx[6]], mp1[rand_idx[7]]])
        rand_mp2 = np.array([mp2[rand_idx[0]], mp2[rand_idx[1]], mp2[rand_idx[2]], mp2[rand_idx[3]], mp2[rand_idx[4]], mp2[rand_idx[5]], mp2[rand_idx[6]], mp2[rand_idx[7]]])

        # Estimate fundamental matrix
        tempF = estimateFundamentalMatrix(rand_mp1, rand_mp2)

        # Find inliers
        inliers = []
        temp_mp1 = []
        temp_mp2 = []
        for j in range(len(mp1)):
            x_1 = mp1[j]
            x_2 = mp2[j]
            error = np.dot(x_2.T, np.dot(tempF, x_1))
            if error < threshold:
                inliers.append(j)
                temp_mp1.append(x_1)
                temp_mp2.append(x_2)

        # Update inliers
        if len(inliers) > inliers_count:
            inliers_count = len(inliers)
            mp1_inlier = temp_mp1
            mp2_inlier = temp_mp2
            new_img_pairs = img_pairs[inliers]

    return np.array(mp1_inlier), np.array(mp2_inlier), np.array(new_img_pairs)