from matplotlib import pyplot as plt
import numpy as np
import heapq
from math import inf, nan
from math import log
from collections import Counter

class Heap:
    def __init__(self, arr=None, key=lambda x: x, max_len=inf):
        self.key = key
        self.max_len = max_len
        if not arr:
            self.h = []
        else:
            self.h = [(self.key(i), i) for i in arr]
        heapq.heapify(self.h)
        self.i = 0

    def __len__(self):
        return len(self.h)

    def __bool__(self):
        return len(self.h) != 0

    def __iter__(self):
        while self:
            yield self.pop()

    def push(self, x):
        # insert an number to the middle so that `x` will be never compared
        # because maybe `x` doesn't have comparing operator defined
        heapq.heappush(self.h, (self.key(x), self.i, x))
        self.i += 1
        if len(self.h) > self.max_len:
            self.pop()

    def top(self):
        return self.h[0][-1]

    def pop(self):
        return heapq.heappop(self.h)[-1]

def argmax(arr, key=lambda x: x):
    arr = [key(a) for a in arr]
    ans = max(arr)
    return arr.index(ans), ans

def argmin(arr, key=lambda x: x):
    arr = [key(a) for a in arr]
    ans = min(arr)
    return arr.index(ans), ans

# ------------------ Decision Trees -------------------------------------------
def entropy(p):
    s = sum(p)
    p = [i / s for i in p]
    ans = sum(-i * log(i) for i in p)
    return ans

def entropy_of_split(X, Y, col):
    """calculate the conditional entropy of splitting data by col"""
    val_cnt = Counter(x[col] for x in X)
    ans = 0
    for val in val_cnt:
        weight = val_cnt[val] / len(X)
        entr = entropy(Counter(y for x, y in zip(X, Y) if x[col] == val).values())
        ans += weight * entr
    return ans

def information_gain(X, Y, col):
    entropy_of_X = entropy(Counter(Y).values())
    entropy_of_col = entropy_of_split(X, Y, col)
    return entropy_of_X - entropy_of_col

def information_gain_ratio(X, Y, col):
    information_gain_of_col = information_gain(X, Y, col)
    entropy_of_col = entropy(Counter(x[col] for x in X).values())
    return information_gain_of_col / entropy_of_col

def gini(Y):
    cnt = Counter(Y)
    ans = 0.
    for y in cnt:
        ans += (cnt[y] / len(Y)) ** 2
    return 1 - ans

# ------------------ Geometry -------------------------------------------------
def kbline(k, b, **args):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + k * x_vals
    plt.plot(x_vals, y_vals, **args)

def wbline(w, b, **args):
    k = -w[0] / w[1]
    b /= -w[1]
    if np.isinf(k):
        plt.vlines(b / w1, plt.gca().get_ylim(), **args)
    else:
        kbline(k, b, **args)

def euc_dis(a, b):
    return np.linalg.norm(a - b)
