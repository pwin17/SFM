import numpy as np

def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    # x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(uv.T)).T

    return  x_norm, T


def estimateFundamentalMatrix(mp1, mp2):
    if(mp1.shape[0]<8 and mp2.shape[0]<8) or (mp1.shape[0]!=mp2.shape[0]):
        return None
    else:
        x1 = mp1
        x2 = mp2
        x1, T1 = normalize(x1)
        x2, T2 = normalize(x2)


        A = np.zeros((mp1.shape[0],9))
        for i in range(len(x1)):
            x_1,y_1 = x1[i,0], x1[i,1]
            x_2,y_2 = x2[i,0], x2[i,1]

            A[i] = [x_2 * x_1, x_2 * y_1, x_2, y_2 * x_1, y_2 * y_1, y_2, x_1, y_1, 1]
        
        # Finding rank 3 F using SVD
        U, D, Vt = np.linalg.svd(A)
        V = Vt.T
        F = V[:,-1].reshape(3,3)

        # Fixing rank of F to 2
        U, D, Vt = np.linalg.svd(F)
        D = np.diag(D)
        D[2,2] = 0
        UD = np.dot(U,D)
        F = np.dot(UD, Vt)
        F = np.dot(T2.T, np.dot(F,T1))
        return F