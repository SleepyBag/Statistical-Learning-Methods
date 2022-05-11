import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import euc_dis

class KMeans:
    def __init__(self, k, max_iterations=1000, verbose=False):
        self.k = k
        self.max_iterations = max_iterations
        self.verbose = verbose

    def fit(self, X):
        """
        X is a matrix shaped of [data_size, feature_size]
        """
        X = X.astype(float)
        data_size, feature_size = X.shape

        self.centers = X[np.random.choice(data_size, self.k, replace=False)]
        self.centers = np.array([[0, 7], [-2, -3]]).astype(float)
        pre_centers = self.centers - 1
        step = 0
        if self.verbose:
            print('Initial centroids:', self.centers)
        while (pre_centers != self.centers).any():
            pre_centers = self.centers.copy()
            # distance from each data sample to the centroid
            # dis[i, j] is the distance from i-th data sample to the j-th centroid
            # shape: [data_size, k]
            dis = euc_dis(X[:, None, :], self.centers[None, :, :])
            # assignment of each data sample to centroid
            # cluster[i] is the index of cluster of i-th data sample
            # shape: [data_size]
            cluster = dis.argmin(axis=-1)
            for i in range(self.k):
                self.centers[i] = X[cluster == i].mean(axis=0)
            step += 1
            if self.verbose:
                print('Step', step)
                print('Assignment:', cluster)
                print('Centroids:', self.centers)
            if step == self.max_iterations:
                break

    def predict(self, X):
        dis = euc_dis(X[:, None, :], self.centers[None, :, :])
        return dis.argmin(axis=-1)

if __name__ == "__main__":
    def demonstrate(X, k, desc):
        k_means = KMeans(k=k, verbose=True)
        k_means.fit(X)
        pred = k_means.predict(X)

        # plot
        plt.scatter(k_means.centers[:, 0], k_means.centers[:,1], marker='x', label='centroids')
        plt.scatter(X[:,0], X[:,1], c=pred, s=20, label='data samples')
        plt.legend()
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
