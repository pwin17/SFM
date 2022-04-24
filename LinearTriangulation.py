import numpy as np

def LinearTriangulation(K, C0, R0, C1, R1, mp1, mp2):
    # Given 2D points, this function computes the 3D point using linear triangulation
    I = np.eye(3)

    C0 = np.hstack((I, -1*C0))
    P0 = np.dot(np.dot(K,R0), C0)

    C1 = C1.reshape(-1,1)
    C1 = np.hstack((I, -1*C1))
    P1 = np.dot(np.dot(K,R1), C1)

    P01, P02, P03 = P0[0,:].T, P0[1,:].T, P0[2,:].T
    P01, P02, P03 = P01.reshape(-1,1), P02.reshape(-1,1), P03.reshape(-1,1)

    P11, P12, P13 = P1[0,:].T, P1[1,:].T, P1[2,:].T
    P11, P12, P13 = P11.reshape(-1,1), P12.reshape(-1,1), P13.reshape(-1,1)

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
        S, U, Vt = np.linalg.svd(A)
        V = Vt.T
        X = V[:, -1]
        world_points.append(X)
    return np.array(world_points)

