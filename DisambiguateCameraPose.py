import numpy as np 

def get_best_pose(pts_3D_all, R, C):
    max_positive_z = 0
    idx = None
    for i in range(len(R)):
        R1 = R[i]
        C1 = C[i]
        r3 = R1[2,:]
        pts_3D = pts_3D_all[i]
        pts_3D = (pts_3D/pts_3D[3,:])[:3,:]
        
        positive_z = get_positive_z_count(pts_3D,C1,r3)
        
        if positive_z > max_positive_z:
            max_positive_z = positive_z
            idx = i
    return R[idx], C[idx], pts_3D_all[idx]

def get_positive_z_count(pts_3D,C,r3):
    positive_z = 0
    for i in range(pts_3D.shape[1]):
        X = pts_3D[:,i]
        val = np.dot(r3, (X - C))
        if val>0:
            positive_z+=1

    return positive_z