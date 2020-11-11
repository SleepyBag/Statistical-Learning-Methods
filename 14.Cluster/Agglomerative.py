import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import euc_dis

class Agglomerative:
    def __init__(self, k):
        self.k = k

    def get_root(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.get_root(self.parent[i])
        return self.parent[i]

    def fit_predict(self, X):
        """
        X is a matrix shaped of [data_size, feature_size]
        """
        data_size, feature_size = X.shape
        self.cluster_num = data_size

        self.parent = [i for i in range(data_size)]
        dis = euc_dis(X[:, None, :], X[None, :, :])
        sorted_a, sorted_b = np.unravel_index(np.argsort(dis, axis=None), dis.shape)
        for a, b in zip(sorted_a, sorted_b):
            root_a, root_b = self.get_root(a), self.get_root(b)
            if root_a != root_b:
                if root_a > root_b:
                    root_a, root_b = root_b, root_a
                self.parent[root_b] = root_a

                self.cluster_num -= 1
                if self.cluster_num == self.k:
                    break

        root = [self.get_root(i) for i in range(data_size)]
        root_map = {n: i for i, n in enumerate(sorted(list(set(root))))}
        return [root_map[r] for r in root]


if __name__ == "__main__":
    def demonstrate(X, k, desc):
        agglomerative = Agglomerative(k=k)
        pred = agglomerative.fit_predict(X)

        # plot
        plt.scatter(X[:,0], X[:,1], c=pred, s=20)
        plt.title(desc)
        plt.show()

    # -------------------------- Example 1 ----------------------------------------
    X = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [2, 1], [1, 2]])
    # generate grid-shaped test data
    demonstrate(X, 2, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.3, .3], [100, 2]),
        np.random.normal([0, 1], [.3, .3], [100, 2]),
        np.random.normal([1, 0], [.3, .3], [100, 2]),
    ])
    # generate grid-shaped test data
    demonstrate(X, 3, "Example 2: it is very sensitive to noise")
