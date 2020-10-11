import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from functools import partial
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

class KNN:
    def __init__(self, k=1, distance_func="l2"):
        self.k = k
        if distance_func == 'l2':
            self.distance_func = lambda x, y: np.linalg.norm(x - y)
        else:
            self.distance_func = distance_func

    def _knn(self, x):
        dis = np.apply_along_axis(partial(self.distance_func, y=x), axis=-1, arr=self.X)
        topk_ind = np.argpartition(dis, self.k)[:self.k]
        return topk_ind

    def _predict(self, x):
        topk_ind = self._knn(x)
        topk_y = self.Y[topk_ind]
        return np.argmax(np.bincount(topk_y))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.k = min(self.k, len(self.X))

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == "__main__":
    def demonstrate(X_train, Y_train, X_test, k, desc):
        console = Console(markup=False)
        knn = KNN(k=k)
        knn.fit(X_train, Y_train)
        pred_test = knn.predict(X_test)

        # plot
        plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, s=20)
        plt.scatter(X_test[:,0], X_test[:,1], c=pred_test, marker=".", s=1)
        plt.title(desc)
        plt.show()

    # -------------------------- Example 1 ----------------------------------------
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 2, 3, 4, 5])
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 1")

    # -------------------------- Example 2 (Imblance Data) ------------------------
    print("Example 2:")
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 1, 2, 3, 4])
    knn = KNN(k=2)
    knn.fit(X_train, Y_train)
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 2")

    # -------------------------- Example 2 (Imblance Data) ------------------------
    print("Example 2:")
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    Y_train = np.array([1, 1, 2, 2, 2])
    knn = KNN(k=3)
    knn.fit(X_train, Y_train)
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    demonstrate(X_train, Y_train, X_test, 1, "Example 2")
