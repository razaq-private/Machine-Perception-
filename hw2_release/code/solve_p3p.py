import numpy as np
import sys

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    a = np.linalg.norm(Pw[1] - Pw[2])
    b = np.linalg.norm(Pw[0] - Pw[2])
    c = np.linalg.norm(Pw[0] - Pw[1])
    print('a, b, c: ', a, b, c)
   
    
    Pc_4x3 = np.hstack((Pc, np.ones((4,1))))
    noCoordinates = np.array([np.linalg.inv(K) @ i for i in Pc_4x3])
    print('noCoord:', noCoordinates)
    #find the pixel coordinates 
    u = []
    v = []
    j = []

    for i in noCoordinates[0:3, :]: 
        u.append(i[0])
        v.append(i[1])
    
    for i in range(0,3): 
        j.append((1/np.sqrt((u[i]**2) + (v[i]**2) + (1**2))) * np.array([u[i], v[i], 1]))

    print('j:',j)

    alpha = np.arccos(np.dot(j[1], j[2]))
    beta = np.arccos(np.dot(j[0], j[2]))
    gamma = np.arccos(np.dot(j[0], j[1]))


    A4 = ((((a ** 2 - c ** 2) / b ** 2) - 1) ** 2) - ((4 * c ** 2) / b ** 2) * (np.cos(alpha)) ** 2

    A3 = 4 * (((a ** 2 - c ** 2) / b ** 2) * (1 - ((a ** 2 - c ** 2) / b ** 2)) * np.cos(beta) - (1 - ((a ** 2 + c ** 2) / b ** 2)) * np.cos(alpha) * np.cos(gamma) + 2 * (c ** 2 / b ** 2) * (np.cos(alpha) ** 2) * np.cos(beta))

    A2 = 2 * (((a ** 2 - c ** 2) / b ** 2) ** 2 - 1 + 2 * ((a ** 2 - c ** 2) / b ** 2) ** 2 * (np.cos(beta)) ** 2 + 2 * ((b ** 2 - c ** 2) / b ** 2) * (np.cos(alpha)) ** 2 - 4 * ((a ** 2 + c ** 2) / b ** 2) * np.cos(alpha) * np.cos(beta) * np.cos(gamma)+ 2 * ((b ** 2 - a ** 2) / b ** 2) * (np.cos(gamma)) ** 2)

    A1 = 4 * (-((a ** 2 - c ** 2) / (b ** 2)) * (1 + (a ** 2 - c ** 2) / (b ** 2)) * np.cos(beta) + (((2 * a ** 2) / b ** 2)) * (np.cos(gamma)) ** 2 * np.cos(beta) - (1 - ((a ** 2 + c ** 2) / b ** 2)) * np.cos(alpha) * np.cos(gamma))

    A0 = (1 + (a ** 2 - c ** 2) / (b ** 2)) ** 2 - ((4 * a ** 2) / (b ** 2)) * ((np.cos(gamma)) ** 2)

    polynomial = [A4, A3, A2, A1, A0]
    
    roots = np.roots(polynomial)
    roots = roots[np.iscomplex(roots) == False] #filter out the complex numbers 
    roots = np.array([r.real for r in roots])

    v_roots = roots[roots > 0]
    u_roots = []
    s1 = []
    s2 = []
    s3 = []

    for v in v_roots: 
        u_roots.append((((-1 + (((a**2) - (c ** 2)) / (b ** 2)))) * (v ** 2) - 2 * ((a**2) - (c ** 2)) / (b ** 2) * np.cos(beta) * v + 1 + (((a**2) - (c ** 2)) / (b ** 2))) / (2 * (np.cos(gamma) - v * np.cos(alpha))))
    
    u_roots = np.array(u_roots)

   
    for u in u_roots: 
        s1.append(np.sqrt((c ** 2) / (1 + (u ** 2) - 2 * u * np.cos(gamma))))

    s1 = np.array(s1)
    s2 = []
    s3 = []
    for i in range(len(s1)):
        s2.append(u_roots[i]*s1[i])
        s3.append(v_roots[i]*s1[i])
    
    s2 = np.array(s2)
    s3 = np.array(s3)

    conditional = np.logical_and(s1 > 0, s2 > 0)
    conditional = np.logical_and(conditional, s3 > 0)

    s1 = s1[conditional]
    s2 = s2[conditional]
    s3 = s3[conditional]
    
    min_dist = sys.maxsize
    R = None
    t = None

    print(s1, s2, s3)

    for i in range(len(s1)):
        possible_PC = np.vstack((s1[i] * j[0], s2[i] * j[1], s3[i] * j[2]))
        R, t = Procrustes(possible_PC, Pw[0:3])
        
        R_cw = np.linalg.inv(R)
        T_cw = -R_cw @ t

        Pc_final = K @ ((R_cw @ Pw[3]) + T_cw)
        
        Pc_final = Pc_final / Pc_final[2]
        dist = np.linalg.norm((Pc_final[0:2] - Pc[3]))
        
        if dist < min_dist:
            min_dist = dist
            print('min dist: ', min_dist)
            R_cond = R
            print('R min_dist:', R_cond)
            t_cond = t
            print('t min_dist:', t)
        
   
  
    
    R = R_cond
    t = t_cond
    print('R:', R)
    print('t:', t)
   
    ##### STUDENT CODE END #####

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    A_centroid = np.average(X, axis = 0)
    B_centroid = np.average(Y, axis = 0)

    A_center = X - A_centroid
    B_center = Y - B_centroid
    
    abt = np.dot(A_center.T, B_center)

    U, s, Vt = np.linalg.svd(abt)
    temp = np.zeros((3,3))
    diag_temp = Vt.T @ U.T
    np.fill_diagonal(temp, [1, 1, np.linalg.det(diag_temp)])

    R = Vt.T @ temp @ U.T
    t = B_centroid - np.dot(R, A_centroid)
    ##### STUDENT CODE END #####

    return R, t
