import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
        X' ~ HX

    """

    ##### STUDENT CODE START #####
    A = []

    for i in range(0,4): 
        A.append([-X[i][0], -X[i][1], -1, 0, 0, 0, X[i][0] * Y[i][0], X[i][1] * Y[i][0], Y[i][0]])
        A.append([0, 0, 0, -X[i][0], -X[i][1], -1, X[i][0] * Y[i][1], X[i][1] * Y[i][1], Y[i][1]])

    u, s, v = np.linalg.svd(A)
    V = v.T
    V = np.reshape(v[-1], (9,1))
    
    H = np.reshape(V, (3,3))

    ##### STUDENT CODE END #####
    
    return H



