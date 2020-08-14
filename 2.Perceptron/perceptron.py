import numpy as np
from rich.console import Console
from rich.table import Table
import sys

class Perceptron:
    def __init__(self, lr=1e-1, max_iteration=2000, verbose=False):
        self.lr = lr
        self.verbose = verbose
        self.max_iteration = max_iteration

    def _trans(self, x):
        return self.w @ x + self.b

    def _predict(self, x):
        return 1 if self._trans(x) >= 0. else -1

    def fit(self, X, Y):
        self.feature_size = X.shape[-1]
        # define parameteres
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)

        updated = 1
        epoch = 0
        # if there is mis-classified sample, train
        while updated > 0 and epoch < self.max_iteration:
            if self.verbose:
                print(f"epoch {epoch} started...")

            updated = 0
            # shuffle data
            perm = np.random.permutation(len(X))
            for i in perm:
                x, y = X[i], Y[i]
                # if there is a mis-classified sample
                if self._predict(x) != y:
                    # update the parameters
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    updated += 1

            if self.verbose:
                print(f"epoch {epoch} finishied, {updated} pieces of data mis-classified")
            epoch += 1
        return

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == "__main__":
    console = Console(markup=False)
    perceptron = Perceptron(verbose=True)
    print("Example 1:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    perceptron.fit(X, Y)

    pred = perceptron.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    print("Example 2: (Perceptron cannot solve a simple XOR problem)")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, -1, 1])
    perceptron.fit(X, Y)

    pred = perceptron.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)
