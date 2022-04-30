import numpy as np

def BuildVisibilityMatrix(flag3D, flag2D, nCams):

    mixed2Dflag = np.zeros((flag2D.shape[0],1), dtype = int)
    rows, columns = np.where(flag2D[:,0:nCams+1] == 1)
    rows = np.array(list(set(rows)))
    mixed2Dflag[rows] = 1

    idx_2D3D = np.where((flag3D.reshape(-1)) & (mixed2Dflag.reshape(-1)))

    for j in range(nCams + 1):
        v_i = flag2D[idx_2D3D, j].reshape(-1,1)
        if j == 0:
            VisibilityMtx = v_i
        else:
            VisibilityMtx = np.hstack((VisibilityMtx, v_i))
    return idx_2D3D, VisibilityMtx