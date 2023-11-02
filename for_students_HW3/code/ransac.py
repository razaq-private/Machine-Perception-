from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8
    eps = 10**-4
    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(
            seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]


        """ YOUR CODE HERE
        """
        # estimate E using the sampled points
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])

        # calculate distances to epipolars for all test points at once
        e_3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        x1_test = X1[test_indices]
        x2_test = X2[test_indices]
        inliers = []

        for i in range(len(test_indices)):
                x1 = x1_test[i]
                x2 = x2_test[i]
                dist = ((x2.T @ E @ x1) ** 2) / (np.linalg.norm(e_3 @ E @ x1) ** 2) + ((x2.T @ E @ x1) ** 2) / (np.linalg.norm(e_3 @ E.T @ x2) ** 2)
                if dist < eps:
                    inliers.append(test_indices[i])

        inliers = np.array(inliers)
        inliers = np.concatenate((sample_indices, inliers))
        print(inliers)
        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers

    return best_E, best_inliers
