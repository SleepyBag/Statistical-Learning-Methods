from matplotlib import pyplot as plt
import numpy as np
import heapq
from math import inf, nan
from math import log, sqrt
from collections import Counter

# ------------------ Basic Structures -----------------------------------------
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

# ------------------ Functions ------------------------------------------------
def argmax(arr, key=lambda x: x):
    arr = [key(a) for a in arr]
    ans = max(arr)
    return arr.index(ans), ans

def argmin(arr, key=lambda x: x):
    arr = [key(a) for a in arr]
    ans = min(arr)
    return arr.index(ans), ans

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def binary_cross_entropy(pred, Y):
    loss = -(Y * np.log(pred) + (1 - Y) * np.log(1 - pred)).sum()
    return loss

def softmax(logits, axis=-1):
    exps = np.exp(logits)
    return exps / exps.sum(axis=axis, keepdims=True)

def line_search(f, l, r, epsilon=1e-6):
    """find the minimum point of a convex function"""
    lrate = (3 - sqrt(5)) / 2
    rrate = (sqrt(5) - 1) / 2
    fll, frr = None, None
    while r - l >= epsilon:
        if fll is None:
            ll = l + (r - l) * lrate
            fll = f(ll)
        if frr is None:
            rr = l + (r - l) * rrate
            frr = f(rr)
        if fll < frr:
            r, rr = rr, ll
            frr, fll = fll, None
        elif fll > frr:
            l, ll = ll, rr
            fll, frr = frr, None
        else:
            l, r = ll, rr
            fll, frr = None, None
    return (l + r) / 2

def newton(f, g, x0, epsilon=1e-6):
    """
    Find the zero point wehre f(x) = 0 of function f
    g(x) is the gradient function of f
    """
    prex = x0
    x = x0 - f(x0) / g(x0)
    while abs(x - prex) > epsilon:
        prex, x = x, x - f(x) / g(x)
    return x

def one_hot(i, size):
    """Given a hot number the tensor size, return the one-hot tensor"""
    ans = np.zeros(size)
    ans[i] = 1
    return ans

def row_echelon(A):
    """
    eliminate a matrix to row echelon form with gaussian elimination
    """
    # convert A to row echolon form
    row_cnt, col_cnt = A.shape
    col = 0
    rank = 0
    # from top to the bottom
    for i in range(row_cnt):
        find = False
        while not find and col < col_cnt:
            # look for the first non-zero value in current column
            for j in range(i, row_cnt):
                if A[j][col] != 0.:
                    if i != j:
                        A[[i, j]] = A[[j, i]]
                    A[i] /= A[i][col]
                    find = True
                    # if non-zero value found, start elimination
                    for k in range(i + 1, row_cnt):
                        A[k] -= A[i] * A[k][col]
                    rank += 1
                    break
            # if not found, check the next column
            else:
                col += 1
        col += 1
    # from bottom to the top
    for i in range(row_cnt - 1, -1, -1):
        # find the first non-zero value and eliminate
        for col in range(col_cnt):
            if A[i][col] != 0.:
                # start elimination
                for k in range(i - 1, -1, -1):
                    A[k] -= A[i] * A[k][col] / A[i][col]
                break
    return A[: rank]

def get_solution_domain(A):
    """
    get a group of linearly independent solutions of Ax=0, which are normalized
    the input A is supposed to be in row echelon form
    """
    row_cnt, col_cnt = A.shape
    A = row_echelon(A)
    col = 0
    nonzero_cols = []
    ans = []
    for i in range(row_cnt):
        while col != col_cnt and A[i][col] == 0.:
            ans.append(one_hot(col, col_cnt))
            for j, j_col in enumerate(nonzero_cols):
                print(j, j_col)
                ans[-1][j_col] = -A[j][col]
            col += 1
        # record the first nonzero value of each row
        nonzero_cols.append(col)
        col += 1

    for col in range(col, col_cnt):
        ans.append(one_hot(col, col_cnt))
        for i, j in enumerate(nonzero_cols):
            ans[-1][j] = -A[i][col]
    if ans:
        ans = np.stack(ans)
        ans /= np.linalg.norm(ans, axis=-1, keepdims=True)
    else:
        ans = np.zeros([0, col_cnt])
    return ans.T

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
    if w[1] == 0:
        plt.vlines(-b / w[0], *plt.gca().get_ylim(), **args)
    else:
        k = -w[0] / w[1]
        b /= -w[1]
        kbline(k, b, **args)

def euc_dis(a, b):
    return np.linalg.norm(a - b, axis=-1)
