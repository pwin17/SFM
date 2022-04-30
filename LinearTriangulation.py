import numpy as np
# http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
def getP(K, R, C):
    P = np.dot(np.dot(K,R), C)
    P1, P2, P3 = P[0,:].T, P[1,:].T, P[2,:].T
    return P1.reshape(-1,1), P2.reshape(-1,1), P3.reshape(-1,1)

def LinearTriangulation(K, C0, R0, C1, R1, mp1, mp2):
    # Given 2D points, this function computes the 3D point using linear triangulation
    I = np.eye(3)

    C0 = C0.reshape(-1,1)
    C0 = np.hstack((I, -1*C0))
    P01, P02, P03 = getP(K, R0, C0)

    C1 = C1.reshape(-1,1)
    C1 = np.hstack((I, -1*C1))
    P11, P12, P13 = getP(K, R1, C1)

    world_points = []
    
    for i in range(len(mp1)):
        mp1x = mp1[i, 0]
        mp1y = mp1[i, 1]
        mp2x = mp2[i, 0]
        mp2y = mp2[i, 1]

        A = np.array([mp1y * P03 - P02, 
                    P01 - mp1x * P03, 
                    mp2y * P13 - P12, 
                    P11 - mp2x * P13])

        A = A.reshape((4,4))
        _, _, Vt = np.linalg.svd(A)
        V = Vt.T
        X = V[:, -1]
        world_points.append(X)
    return np.array(world_points)

