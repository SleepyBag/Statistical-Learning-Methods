from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table
import numpy as np

class NaiveBayesMLE:
    def __init__(self, verbose=False):
        # p(a|y), the probability of an attribute a when the data is of label y
        # its a three-layer dict
        # the first-layer key is y, the value label
        # the second-layer key is n, which means the nth attribute
        # the thrid-layer key is the value of the nth attribute
        self.pa_y = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        # p(y), the prior probability of label y
        self.py = defaultdict(lambda: 0)
        self.verbose = verbose

    def fit(self, X, Y):
        y_cnt = Counter(Y)
        for x, y in zip(X, Y):
            for i, a in enumerate(x):
                self.pa_y[y][i][a] += 1 / y_cnt[y]
            self.py[y] += 1 / len(X)

        if self.verbose:
            for y in self.pa_y:
                print(f'The prior probability of label {y} is', self.py[y])
                for nth in self.pa_y[y]:
                    prob = self.pa_y[y][nth]
                    for a in prob:
                        print(f'When the label is {y}, the probability that {nth}th attribute be {a} is {prob[a]}')

    def _predict(self, x):
        # all the labels
        labels = list(self.pa_y.keys())
        probs = []
        for y in labels:
            prob = self.py[y]
            for i, a in enumerate(x):
                prob *= self.pa_y[y][i][a]
            probs.append(prob)
        if self.verbose:
            for y, p in zip(labels, probs):
                print(f'The likelihood {x} belongs to {y} is {p}')
        return labels[np.argmax(probs)]

    def predict(self, X):
        return [self._predict(x) for x in X]

if __name__ == "__main__":
    console = Console(markup=False)
    naive_bayes_mle = NaiveBayesMLE(verbose=True)
    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = [
        [1,'S'],
        [1,'M'],
        [1,'M'],
        [1,'S'],
        [1,'S'],
        [2,'S'],
        [2,'M'],
        [2,'M'],
        [2,'L'],
        [2,'L'],
        [3,'L'],
        [3,'M'],
        [3,'M'],
        [3,'L'],
        [3,'L'],
    ]
    Y = [-1 ,-1 ,1 ,1 ,-1 ,-1 ,-1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,-1]
    naive_bayes_mle.fit(X, Y)

    # show in table
    pred = naive_bayes_mle.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)
