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
    def __init__(self):
        pass

    def fit(self, X, Y):
        # since X is one-dimensional, just flatten it
        X = X[:, 0]
        possible_thresholds = list(set(X))
        possible_thresholds.append(max(possible_thresholds) + 1)
        possible_thresholds.append(min(possible_thresholds) - 1)
        # try all possible threshold
        best_acc = 0.
        for self.sign in [1, -1]:
            for self.threshold in possible_thresholds:
                pred = self.predict(X)
                acc = (pred == Y).mean()
                if acc > best_acc:
                    best_acc, best_threshold, best_sign = acc, self.threshold, self.sign
        self.threshold, self.sign = best_threshold, best_sign

    def predict(self, X):
        X = X * self.sign
        pred = X > self.threshold
        return pred

class AdaBoost:
    def __init__(basic_model=DecisionStump, steps=10):
