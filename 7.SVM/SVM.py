from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import wbline

class SVM:
    def __init__(self, C=1e9, epsilon=1e-6, lr=1e-4, max_steps=1000, verbose=True):
        self.lr = lr
        self.max_steps = max_steps
        self.verbose = verbose
        self.C = C
        self.epsilon = epsilon

    def _dual_object(self):
        return .5 * (self.alpha[:, None] * self.alpha * self.Y[:, None] * self.Y * self.K).sum() - self.alpha.sum()

    def _smo_objective(self, i, j):
        """
        The objective function of one step of SMO
        given the choosed alpha i and alpha j
        """
        alpha, Y, K = self.alpha, self.Y, self.K
        return (alpha[i] * Y[i] * alpha * K[i:] * Y).sum() \
            + (alpha[j] * Y[j] * alpha * K[j:] * Y).sum() \
            - .5 * alpha[i] ** 2 * K[i, i] \
            - .5 * alpha[j] ** 2 * K[j, j] \
            - alpha[i] * alpha[j] * Y[i] * Y[j] * K[i, j]\
            - alpha[i] - alpha[j]

    def _smo_step(self):
        if self.verbose:
            print('SMO step start...')
        alpha = self.alpha
        K = self.K
        data_size = len(alpha)

        # the prediction of this step
        pred = (self.alpha * Y * self.K).sum(axis=-1) + self.b
        # the score of pred
        score = Y * pred
        # discrepency between pred and label
        error = pred - Y

        updated = False

        # find the first variable alpha_i
        # which violate KKT constraint
        # first try to find fake support vectors
        # of which 0 < alpha_i < C but score_i isn't 1
        i_cands = [i for i in range(data_size) if
                   0 < alpha[i] < self.C and abs(score[i] - 1) > self.epsilon or
                   alpha[i] == 0 and score[i] < 1 or
                   alpha[i] == self.C and score[i] > 1]
        for i in i_cands:
            # find the second variable
            # which makes alpha_i change most
            relative_error = np.abs(error - error[i])
            j_cands = sorted(list(range(data_size)), key=relative_error.__getitem__)
            for j in j_cands:
                if j == i:
                    continue
                smo_objective_before = self._smo_objective(i, j)

                # upper bound and lower bound of alpha_j
                L = max(0, alpha[j] - alpha[i] if Y[i] != Y[j] else alpha[i] + alpha[j] - self.C)
                H = min(self.C, self.C + alpha[j] - alpha[i] if Y[i] != Y[j] else alpha[i] + alpha[j])

                if self.verbose:
                    print('SMO chooses: ', i, j)
                    print("L and H:", L, H)
                    print('Step begin, current object of dual problem:', smo_objective_before)

                alpha_j_old = alpha[j]
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
                smo_objective_after = self._smo_objective(i, j)
                if self.verbose:
                    print('Step end, current object of dual problem:', smo_objective_after)
                if smo_objective_before - smo_objective_after > self.epsilon:
                    updated = True
                    break
            if updated:
                break
        if self.verbose:
            print('SMO step end...')
            print()
        return len(i_cands) > 0

    def fit(self, X, Y, kernel=lambda x: x @ x.T):
        """
        optimize SVM with SMO
        X: of shape [data-size, feature-size]
        Y: of shape [data-size]
        kernel: kernel function, which
                input is X of shape [data-size, feature-size] and
                output a kernel matrix of shape [data-size, data-size]
        """
        self.X, self.Y = X, Y
        data_size = len(X)
        self.alpha = np.zeros(data_size)
        self.b = np.random.rand()

        self.K = kernel(X)
        # optimize
        while self._smo_step():
            pass

        # optimized, get w and b
        self.w = ((self.alpha * Y)[:, None] * X).sum(axis=0)
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

    # -------------------------- Example 2 ----------------------------------------
    print("Example 2:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    demonstrate(X, Y, "Example 2")

    # -------------------------- Example 3 ----------------------------------------
    print("Example 3:")
    X = np.concatenate((np.random.rand(5, 2), np.random.rand(5, 2) + np.array([1, 1])), axis=0)
    Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    print(X, Y)
    demonstrate(X, Y, "Example 3")
