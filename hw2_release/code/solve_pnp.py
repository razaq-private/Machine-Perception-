from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    #get the homograph matrix but only get the (x,y) of Pw
    Pw_new = Pw[:, 0:2]
    H = est_homography(Pw_new, Pc) 
    H_prime = np.linalg.inv(K) @ H
      
    #step 3: 
    a = H_prime[:, 0]
    b = H_prime[:, 1]
    c = H_prime[:, 2]
    axb = np.cross(a,b)
    axb = np.reshape(axb, (3,1))
    H_prime_fix = np.hstack((H_prime[:, 0:2], axb))

    U, s, Vt = np.linalg.svd(H_prime_fix)
    identity = np.eye(3)
    identity[2][2] = np.linalg.det(U@Vt)
    R = U @ identity @ Vt 
    t = c / np.linalg.norm(a)
    R = R.T
    t = -R @ t


    ##### STUDENT CODE END #####

    return R, t
