import numpy as np

def get_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    C1 = U[:,2]
    R1 = np.dot(U, np.dot(W, Vt))
    if(np.linalg.det(R1)<0):
        R1 = -R1
        C1 = -C1

    C2 = -U[:,2]
    R2 = np.dot(U, np.dot(W, Vt))
    if(np.linalg.det(R2)<0):
        R2 = -R2
        C2 = -C2

    C3 = U[:,2]
    R3 = np.dot(U, np.dot(W.T, Vt))
    if(np.linalg.det(R3)<0):
        R3 = -R3
        C3 = -C3

    C4 = -U[:,2]
    R4 = np.dot(U, np.dot(W.T, Vt))
    if(np.linalg.det(R4)<0):
        R4 = -R4
        C4 = -C4

    C = [C1, C2, C3, C4]
    R = [R1, R2, R3, R4]

    return R, C