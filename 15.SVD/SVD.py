import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

def svd(A):
    """
    given an m x n matrix,
    return the result of SVD,
    as a tuple of (U, Sigma, V)
    """
    m , n = A.shape

    symmetry = A.T @ A
    rank = np.linalg.matrix_rank(symmetry)
    eigen_values, eigen_vectors = np.linalg.eig(symmetry)
    eigen_order = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[eigen_order]

    eigen_values = eigen_values[: rank]
    eigen_vectors = eigen_vectors[:, eigen_order]
    # V is of shape [n, n]
    V = eigen_vectors
    eigen_vectors = eigen_vectors[:, : rank]

    singular_values = np.sqrt(eigen_values)
    singular_matrix = np.zeros_like(A)
    for i, v in enumerate(singular_values):
        singular_matrix[i][i] = v

    U1 = A @ eigen_vectors / singular_values
    U2 = get_solution_domain(row_echelon(A.T))
    U = np.concatenate([U1, U2], axis=-1)
    return U, singular_matrix, V


if __name__ == '__main__':
    def demonstrate(A, desc):
        print(desc)
        U, singular_matrix, V = svd(A)
        print("U is:")
        print(U)
        print("Singular matrix is:")
        print(singular_matrix)
        print("V is:")
        print(V)
        print("The reconstructed matrix is:")
        print(U @ singular_matrix @ V.T)

    A = np.array([[1, 1],
                  [2, 2],
                  [0, 0]]).astype(float)
    demonstrate(A, 'Example 1')

    A = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 4],
                  [0, 3, 0, 0],
                  [0, 0, 0, 0],
                  [2, 0, 0, 0]]).astype(float)
    demonstrate(A, 'Example 2')

    A = np.array([[3, 1],
                  [2, 1]]).astype(float)
    demonstrate(A, 'Example 3')
