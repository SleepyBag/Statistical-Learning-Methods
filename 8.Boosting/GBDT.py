from collections import defaultdict
import numpy as np
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from functools import partial
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import line_search
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent / '5.DecisionTree'))
from RegressionCART import RegressionCART

class GBDT:
    def __init__(self,
                 loss_function=lambda label, pred: ((label - pred) ** 2).sum(),
                 gradient_function=lambda label, pred: 2 * (pred - label),
                 steps=10,
                 max_depth=3,
                 verbose=True):
        """
        `loss_function` takes two arguments, label and pred and return a scalar, the loss
        `gradient_function` is gradient from loss function to the prediction
        It takes two arguments, i.e., label and pred and return the gradient
        the loss function should be convex
        The default loss function is l2 loss, which makes GBDT an ordinary boosting tree
        """
        self.steps = steps
        self.verbose = verbose
        self.gradient_function = gradient_function
        self.loss_function = loss_function
        self.max_depth = max_depth

    def _loss_of_const(self, Y, c):
        """
        Return the loss when the model take a constant c as the prediction
        `Y` is a vector of labels
        `c` is a constant scalar
        """
        c = (np.ones_like(Y) * c).astype(float)
        return self.loss_function(Y, c)

    def fit(self, X, Y):
        n = len(X)
        self.carts = []
        # the basic value of prediction, so that there can be 'residual'
        self.basic_pred = line_search(partial(self._loss_of_const, Y), min(Y), max(Y))

        cur_pred = np.zeros_like(Y) + self.basic_pred
        residual = -self.gradient_function(Y, cur_pred)
        for i in range(self.steps):
            if self.verbose:
                print(f'step {i}')
                print(f'Current pred is {cur_pred}')
                print(f'Current residual is {residual}')
            cart = RegressionCART(verbose=False, max_depth=self.max_depth)
            cart.fit(X, residual)
            self.carts.append(cart)
            # regression trees use l2 loss as loss function,
            # the return value leaf nodes should be recorrect
            leaf2label=defaultdict(list)
            for i, x in enumerate(X):
                leaf = cart._query_leaf(cart.root, x)
                leaf2label[leaf].append(i)
            for leaf in leaf2label:
                data_ind = np.stack(leaf2label[leaf])
                leafY = Y[data_ind]
                leaf_cur_pred = cur_pred[data_ind]
                leaf.label = line_search(lambda c: self.loss_function(leafY, leaf_cur_pred + c), -1e9, 1e9)

            # update the incremental prediction
            inc_pred = cart.predict(X)
            cur_pred += inc_pred
            residual = -self.gradient_function(Y, cur_pred)

    def predict(self, X):
        pred = np.zeros(len(X)) + self.basic_pred
        for cart in self.carts:
            pred += cart.predict(X)
        return pred

if __name__ == "__main__":
    def demonstrate(X, Y, max_depth, desc):
        print(desc)
        console = Console(markup=False)
        gbdt = GBDT(verbose=True, max_depth=max_depth)
        gbdt.fit(X, Y)

        # show in table
        pred = gbdt.predict(X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    # -------------------------- Example 1 ----------------------------------------
    X = np.arange(10).reshape(-1, 1)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    demonstrate(X, Y, 3, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    X = np.arange(10).reshape(-1, 1)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    demonstrate(X, Y, 1, "Example 2: CART cannot be all stumps")

    # -------------------------- Example 3 ----------------------------------------
    X = np.arange(10).reshape(-1, 1)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    demonstrate(X, Y, 2, "Example 3")
