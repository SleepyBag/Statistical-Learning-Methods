from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import argmax, argmin, wbline

class SVM:
    def __init__(self, C=1e9, epsilon=1e-6, lr=1e-4, max_steps=1000, verbose=True):
        self.lr = lr
        self.max_steps = max_steps
        self.verbose = verbose
        self.C = C
        self.epsilon = epsilon

    def _smo_step(self, alpha, score, error, K):
        # find the first variable alpha_i
        # which violate KKT constraint
        # first try to find fake support vectors
        data_size = len(alpha)
        for i in range(data_size):
            if abs(0 < alpha[i] < self.C and score[i] - 1) > self.epsilon:
                break
        else:
            # if no fake support vector found, find others
            for i in range(data_size):
                # alpha[i] == 0 means i should be classified correctly
                # alpha[i] == C means i should be classified wrongly
                if alpha[i] == 0 and score[i] < 1 \
                   or alpha[i] == self.C and score[i] > 1:
                    break
            else:
                return False
        # find the second variable
        # which makes alpha_i change most
        relative_error = np.abs(error - error[i])
        j = argmax(relative_error)[0]
        # TODO: don't choose j randomly
        if j == i:
            j = (i + 1) % data_size
        j = i
        while j == i:
            j = np.random.randint(0, data_size)
        print('alpha', alpha)
        print('score', score)
        print('error', error)
        print('relative error', relative_error)
        print("smo selects ", i, j)

        alpha_j_old = alpha[j]
        # upper bound and lower bound of alpha_j
        L = max(0, alpha[j] - alpha[i] if Y[i] != Y[j] else alpha[i] + alpha[j] - self.C)
        H = min(self.C, self.C + alpha[j] - alpha[i] if Y[i] != Y[j] else alpha[i] + alpha[j])
        print("L and H:", L, H)
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        # update alpha_j
        alpha[j] += Y[j] * (error[i] - error[j]) / eta
        # clip
        alpha[j] = min(alpha[j], H)
        alpha[j] = max(alpha[j], L)
        # update alpha_i
        alpha[i] += Y[i] * Y[j] * (alpha_j_old - alpha[j])
        # update b
        self.b = Y[i] - (alpha * Y * K[i]).sum()
        if 0 < alpha[j] < self.C:
            self.b = (Y[j] - (alpha * Y * K[j]).sum() + self.b) / 2
        return True

    def fit(self, X, Y, kernel=None):
        """
        optimize SVM with SMO
        X: of shape [data-size, feature-size]
        Y: of shape [data-size]
        kernel: kernel function, which
                input is X of shape [data-size, feature-size] and
                output a kernel matrix of shape [data-size, data-size]
        """
        data_size = len(X)
        alpha = np.zeros(data_size)
        self.b = np.random.rand()

        if kernel is None:
            def kernel(X):
                return X @ X.T

        K = kernel(X)
        # optimize
        while True:
            # the prediction of this step
            pred = (alpha * Y * K).sum(axis=-1) + self.b
            # the score of pred
            score = Y * pred
            # discrepency between pred and label
            error = pred - Y

            # update
            if not self._smo_step(alpha, score, error, K):
                break

        # optimized, get w and b
        self.w = ((alpha * Y)[:, None] * X).sum(axis=0)
        # find a support vector, because it determines b
        for i in range(data_size):
            if 0 < alpha[i] < self.C:
                break
        if self.verbose:
            print("Done!")

    def predict(self, X):
        score = (self.w * X).sum(axis=-1) + self.b
        pred = (score >= 0).astype(int) * 2 - 1
        return pred

if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        console = Console(markup=False)
        svm = SVM(verbose=True)
        svm.fit(X, Y)

        # plot
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        wbline(svm.w, svm.b)
        plt.title(desc)
        plt.show()

        # show in table
        pred = svm.predict(X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.array([[0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, -1])
    demonstrate(X, Y, "Example 1")

    # # -------------------------- Example 2 ----------------------------------------
    # print("Example 2:")
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Y = np.array([1, 1, -1, -1])
    # demonstrate(X, Y, "Example 2")

    # -------------------------- Example 2 ----------------------------------------
    print("Example 3:")
    X = np.concatenate((np.random.rand(5, 2), np.random.rand(5, 2) + np.array([1, 1])), axis=0)
    Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    print(X, Y)
    demonstrate(X, Y, "Example 3")

    # # -------------------------- Example 2 ----------------------------------------
    # print("Example 2: SVM cannot solve a simple XOR problem")
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Y = np.array([1, -1, -1, 1])
    # demonstrate(X, Y, "Example 2: SVM cannot solve a simple XOR problem")
