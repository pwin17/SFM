import numpy as np
import cv2

def VisibilityMatrix(mp_1, mp_2, img_pairs):

    VisibilityMtx = np.zeros((6, img_pairs.shape[0]))
    for j in range(img_pairs.shape[0]):
        i1, i2 = img_pairs[j]
        for i in range(6):
            if i+1==int(i1):
                
                VisibilityMtx[i,j]=1
            elif i+1 == int(i2):
                VisibilityMtx[i,j]=1

    return VisibilityMtx

    

