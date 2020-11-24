import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent / '15.SVD'))
from SVD import svd

def pca(X, k=5):
    """
    given a normlized matrix X, each of whose column is a sample
    the dimension of the principle component, k
    return the principle component matrix
    """
    m, n = X.shape
    X_trans = 1 / sqrt(n - 1) * X.T
    _, _, V = svd(X_trans)
    V = V[:, :k]
    return V.T @ X

if __name__ == '__main__':
    def demonstrate(X, k, desc):
        print(desc)
        X -= X.mean(axis=-1, keepdims=True)
        X_trans = pca(X, k=k)
        print(X_trans)

    X = np.array([[1, 1],
                  [2, 2],
                  [0, 0]]).astype(float)
    demonstrate(X, 1, 'Example 1')

    X = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 4],
                  [0, 3, 0, 0],
                  [0, 0, 0, 0],
                  [2, 0, 0, 0]]).astype(float)
    demonstrate(X, 1, 'Example 2')

    X = np.array([[3, 1],
                  [2, 1]]).astype(float)
    demonstrate(X, 1, 'Example 3')

    X = np.array([[0, 0],
                  [-1, 1]]).astype(float)
    demonstrate(X, 1, 'Example 3')
