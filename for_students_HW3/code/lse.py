import numpy as np


def least_squares_estimation(X1, X2):
    """ YOUR CODE HERE
    """
    # take each row of x1 and kronecker product with x2's row
    n = X1.shape[0]
    X = np.zeros((n, 9))
    for i in range(n):
        X[i] = np.outer(X1[i], X2[i]).reshape(-1)
  
    U, _, Vt = np.linalg.svd(X)
    E = Vt[-1].reshape(3, 3).T
    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vt
    

    """ END YOUR CODE
    """
    return E