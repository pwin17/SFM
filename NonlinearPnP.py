import numpy as np
from LinearPnP import *
from scipy.optimize import least_squares
from LinearTriangulation import getP

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def quaternion_rotation_matrix(q0, q1, q2, q3):  
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])                        
    return rot_matrix

def reprojection_error(qx, qy, qz, qw, C, K, X, x):
    I = np.eye(3)
    C = np.hstack((I, -1*C))
    R = quaternion_rotation_matrix(qx, qy, qz, qw)
    P1, P2, P3 = getP(K, R, C)

    u1_reproj = np.divide(np.dot(X, P1), np.dot(X, P3))
    v1_reproj = np.divide(np.dot(X, P2), np.dot(X, P3))
    error = np.square(x[0] - u1_reproj) + np.square(x[1] - v1_reproj)
    return error

def NonlinearPnp(X, x, K, C, R):

    alpha, beta, gamma = rot2eul(R)
    qx, qy, qz, qw = get_quaternion_from_euler(alpha, beta, gamma)
    # for i in range(len(X)):
        # print('shapes: ', worldpts[i].shape, mp1[i].shape, mp2[i].shape)
    optimized_params = least_squares(reprojection_error, [qx, qy, qz, qw, C], args=(K, X, x))
    optimized_WP = optimized_params.x

    [qx, qy, qz, qw, C] = optimized_WP
    R = quaternion_rotation_matrix(qx, qy, qz, qw)

    return R, C
    
