import numpy as np


def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    Pw = []
    pixels = np.hstack((np.array(pixels), np.ones((len(pixels), 1))))
    for p in pixels:
        temp = R_wc @ np.linalg.inv(K) @ p
        Pw.append(-t_wc[2] / temp[2] * temp + t_wc)
    Pw = np.array(Pw)

    ##### STUDENT CODE END #####
    return Pw
