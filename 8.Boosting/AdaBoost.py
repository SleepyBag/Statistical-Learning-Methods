from math import log
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from functools import partial
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import wbline

class DecisionStump:
    """
    A simple classifier.
    A decision stump divide dataset by a threshold
    Expected one-dimensional X
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, X, Y, weight):
        # since X is one-dimensional, just flatten it
        X = X[:, 0]
        possible_thresholds = list(set(X))
        possible_thresholds.append(max(possible_thresholds) + 1)
        possible_thresholds.append(min(possible_thresholds) - 1)
        # try all possible threshold
        best_acc = 0.
        best_threshold, best_sign = 0., 0.
        for self.sign in [1, -1]:
            for self.threshold in possible_thresholds:
                pred = self.predict(X)
                acc = (pred == Y) @ weight
                if acc > best_acc:
                    best_acc, best_threshold, best_sign = acc, self.threshold, self.sign
        self.threshold, self.sign = best_threshold, best_sign
        if self.verbose:
            print(f'Threshold is {self.threshold}')

    def predict(self, X):
        X = X * self.sign
        threshold = self.threshold * self.sign
        pred = (X > threshold) * 2 - 1
        return pred.flatten()

class AdaBoost:
    def __init__(self, BasicModel=DecisionStump, steps=10, verbose=True):
        self.BasicModel = BasicModel
        self.steps = steps
        self.verbose = verbose

    def fit(self, X, Y):
        n = len(X)
        weight = np.ones(n) / n
        self.basic_models = []
        self.model_weights = []
        for i in range(self.steps):
            basic_model = self.BasicModel()
            basic_model.fit(X, Y, weight)
            self.basic_models.append(basic_model)
            pred = basic_model.predict(X)
            error_rate = (pred != Y) @ weight
            model_weight = .5 * log((1 - error_rate) / error_rate)
            weight *= np.exp(-model_weight * Y * pred)
            weight /= weight.sum()
            self.model_weights.append(model_weight)
            if self.verbose:
                print(f'Step {i}, current error rate is {error_rate}')
                print(f'The weight of current model is {model_weight}')

    def predict(self, X):
        score = sum(model.predict(X) * weight for model, weight in
                    zip(self.basic_models, self.model_weights))
        pred = (score > 0.).astype(int) * 2 - 1
        return pred

if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        console = Console(markup=False)
        adaboost = AdaBoost(verbose=True)
        adaboost.fit(X, Y)

        # show in table
        pred = adaboost.predict(X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.arange(10).reshape(-1, 1)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    demonstrate(X, Y, "Example 1")
