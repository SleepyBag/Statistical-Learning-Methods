import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import sigmoid

class LogisticRegression:
    def __init__(self, lr=1e-4):
        self.lr = lr

    def _loss(self, X, Y):
        pred = self.predict(X)

    def fit(self, X, Y):
        self.feature_size = X.shape()[-1]
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)

    def predcit(self, X):
        logit = self.w @ X.transpose() + self.b
        p = sigmoid(logit)
        Y = (p > .5).astype(int)
        return Y
