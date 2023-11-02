import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    # You should Complete est_homography first!
    H = est_homography(X, Y)

    # ##### STUDENT CODE START #####
    warped_pts = [] 
    
    for point in interior_pts: 
        row = [(H[0][0]*point[0] + H[0][1]*point[1]+ H[0][2])/ (H[2][0]*point[0] + H[2][1]*point[1] + H[2][2]), (H[1][0]*point[0] + H[1][1]*point[1] + H[1][2])/(H[2][0]*point[0] + H[2][1]*point[1] + H[2][2])]
        warped_pts.append(row)

    warped_pts = np.array(warped_pts)
    
    # interiorNew = interior_pts.T.append(np.ones)
    # interiorNew = np.vstack((interior_pts.T, np.ones((1, len(interior_pts)))))
    # warped_pts = np.dot(interiorNew.T, H)
    # warped_pts = warped_pts[:,:2] / warped_pts[:,:-1]
    
    # ##### STUDENT CODE END #####


    return warped_pts
