import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  transform_candidates = []
  # Perform SVD on E
  U, _, Vt = np.linalg.svd(E)

  T1 = U[:, 2]
  T2 = -U[:, 2]


  R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
  neg_R90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
  R1 = U @ R90.T @ Vt
  R2 = U @ neg_R90.T @ Vt

  transform_candidates.append({"T": T1, "R": R1})
  transform_candidates.append({"T": T1, "R": R2})
  transform_candidates.append({"T": T2, "R": R1})
  transform_candidates.append({"T": T2, "R": R2})

  """ END YOUR CODE
  """
  return transform_candidates
