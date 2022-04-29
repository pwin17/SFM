import numpy as np

def BuildVisibilityMatrix(img_pairs, num_imgs):

    VisibilityMtx = np.zeros((img_pairs.shape[0], num_imgs))
    for j in range(img_pairs.shape[0]):
        i1, i2 = img_pairs[j]
        for i in range(num_imgs):
            if i+1==int(i1):
                VisibilityMtx[j,i]=1
            elif i+1 == int(i2):
                VisibilityMtx[j,i]=1
    return VisibilityMtx

def getVisibilityMatrix(fullVisibilityMatrix, rows):
    return fullVisibilityMatrix[rows,:]