import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import euc_dis

class KMeans:
    def __init__(self, k, max_iterations=1000):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        """
        X is a matrix shaped of [data_size, feature_size]
        """
        data_size, feature_size = X.shape

        self.centers = X[np.random.choice(data_size, self.k, replace=False)]
        pre_centers = self.centers - 1
        step = 0
        while (pre_centers != self.centers).any():
            pre_centers = self.centers.copy()
            dis = euc_dis(X[:, None, :], self.centers[None, :, :])
            cluster = dis.argmin(axis=-1)
            for i in range(self.k):
                self.centers[i] = X[cluster == i].mean(axis=0)
            step += 1
            if step == self.max_iterations:
                break

    def predict(self, X):
        dis = euc_dis(X[:, None, :], self.centers[None, :, :])
        return dis.argmin(axis=-1)

if __name__ == "__main__":
    def demonstrate(X, k, desc):
        k_means = KMeans(k=k)
        k_means.fit(X)
        pred = k_means.predict(X)

        # plot
        plt.scatter(X[:,0], X[:,1], c=pred, s=20)
        plt.title(desc)
        plt.show()

    # -------------------------- Example 1 ----------------------------------------
    X = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [2, 1], [1, 2]]).astype(float)
    demonstrate(X, 2, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.3, .3], [100, 2]),
        np.random.normal([0, 1], [.3, .3], [100, 2]),
        np.random.normal([1, 0], [.3, .3], [100, 2]),
    ]).astype(float)
    demonstrate(X, 3, "Example 2")

    # -------------------------- Example 3 ----------------------------------------
    X = np.array([[0, 0], [0, 1], [0, 3]]).astype(float)
    demonstrate(X, 2, "Example 3: K-Means doesn't always return the best answer. (try to run multiple times!)")
