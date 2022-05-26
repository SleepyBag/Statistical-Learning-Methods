import numpy as np
from pprint import pprint
from collections import Counter
from rich.console import Console
from rich.table import Table
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *
from ClassificationCART import ClassificationCART

class PrunedCART:
    def __init__(self, cart, X, Y, val_X, val_Y, verbose=True):
        self.root = cart.root
        self.possible_alpha = {0}
        self.verbose = verbose
        self.prune(self.root, X, Y)
        if self.verbose:
            print("All the possible alpha values are", self.possible_alpha)
        self.alpha = self.choose_alpha(val_X, val_Y, self.possible_alpha)
        if self.verbose:
            print("The best alpha value is", self.alpha)

    def prune(self, root, X, Y):
        """
        prune a classification CART recursively. alpha is the weight of tree size in the loss function
        tag all the nodes with a float alpha
        meaning that if the weight of size of tree in the loss function is larger than alpha,
        this node will be trimmed

        possible_alpha is a empty set, in which this function will insert all the possible alpha value.
        reutrn the loss of all the leaf nodes, and the size of the subtree
        """
        # calculate the gini index of this subtree if the children of root is trimmed
        pruned_gini = len(X) * gini(Counter(Y).values())
        pruned_loss = pruned_gini
        # if root is a leaf node, return loss directly
        if root.col is None:
            return pruned_loss, 1

        # cur_loss record the loss function when root is not trimmed
        cur_loss = 0.
        # size record the size of this subtree
        size = 1

        selected_ind = X[:, root.col] == root.val
        other_ind = X[:, root.col] != root.val
        selected_X = X[selected_ind]
        other_X = X[other_ind]
        selected_Y = Y[selected_ind]
        other_Y = Y[other_ind]

        # trim the left node recursively
        child_loss, child_size = self.prune(root.left, selected_X, selected_Y)
        cur_loss += child_loss
        size += child_size

        # trim the right node recursively
        child_loss, child_size = self.prune(root.right, other_X, other_Y)
        cur_loss += child_loss
        size += child_size

        # alpha means that
        # if the weight of size of tree in the loss function is larger than alpha,
        # this node will be trimmed
        alpha = (pruned_loss - cur_loss) / (size - 1)
        root.alpha = alpha
        self.possible_alpha.add(alpha)
        return cur_loss, size

    def query(self, root, x, alpha):
        # if root.alpha is less than choosed alpha, it is trimmed
        if root.col is None or root.alpha < alpha:
            return root.label
        elif x[root.col] != root.val:
            return self.query(root.right, x, alpha)
        return self.query(root.left, x, alpha)

    def _predict(self, x, alpha):
        return self.query(self.root, x, alpha)

    def predict(self, X, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return np.array([self._predict(x, alpha) for x in X])

    def validate(self, val_X, val_Y, alpha):
        """
        I don't think using gini index for validation, as written in the book, is a good idea,
        beacause gini index is unsupervised but there is label available in the validation set.
        So I choose to use accuracy instead.
        """
        pred = self.predict(val_X, alpha)
        return (pred == val_Y).mean()

    def choose_alpha(self, val_X, val_Y, possible_alpha):
        """
        Choose the best subtree according to the validation set.
        Cross-validation here simply refers to predict on a pre-split validation set.
        """
        best_acc = -1.
        best_alpha = 0.
        for alpha in possible_alpha:
            cur_acc = self.validate(val_X, val_Y, alpha)
            if self.verbose:
                print(f"When alpha = {alpha}, accuracy is {cur_acc}")
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_alpha = alpha
        return best_alpha


if __name__ == "__main__":
    console = Console(markup=False)
    cart = ClassificationCART(verbose=True)
    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.array([
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ])
    Y = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])
    cart.fit(X, Y)

    # Here I use the same dataset as the validation set
    # Notice that it must be the full tree to be choosed this way
    testX = np.array([
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ])
    testY = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])

    pruned_cart = PrunedCART(cart, X, Y, testX, testY)

    # show in table
    pred = pruned_cart.predict(testX)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(testX, testY, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)
