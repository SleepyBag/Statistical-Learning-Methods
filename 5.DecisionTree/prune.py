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
from ID3 import ID3

def prune(root, X, Y, alpha=.0, verbose=True):
    """
    prune a decision tree recursively. alpha is the weight of tree size in the loss function
    reutrn the loss of all the leaf nodes
    """
    pruned_entropy = len(X) * entropy(Counter(Y).values())
    pruned_loss = pruned_entropy + alpha
    # if root is a leaf node, return loss directly
    if not root.children:
        return pruned_loss
    cur_loss = 0.
    for col_val in root.children:
        child = root.children[col_val]
        ind = [x[root.col] == col_val for x in X]
        childX = [x for i, x in zip(ind, X) if i]
        childY = [y for i, y in zip(ind, Y) if i]
        cur_loss += prune(child, childX, childY, alpha, verbose)
    # if pruned, return the pruned loss
    if verbose:
        pprint(X)
        print('loss if prune:', pruned_loss)
        print('current loss', cur_loss)
    if pruned_loss < cur_loss:
        root.children.clear()
        return pruned_loss
    # if not pruned, the loss of node root is the sum loss of all of its children
    return cur_loss


if __name__ == "__main__":
    console = Console(markup=False)
    # -------------------------- Example 1 (Small Normalization Param) ------------
    print("Example 1:")
    id3 = ID3(verbose=False)
    X = [
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
    ]
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    id3.fit(X, Y)

    # prune with alpha 0.
    prune(id3.root, X, Y, 0.)

    # show in table
    pred = id3.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    # -------------------------- Example 2 (Large Normalization Param) ------------
    print("Example 2:")
    id3 = ID3(verbose=False)
    X = [
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
    ]
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    id3.fit(X, Y)

    # prune with large alpha
    prune(id3.root, X, Y, 10000.)

    # show in table
    pred = id3.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)

    # -------------------------- Example 3 (Midium Normalization Param) -----------
    print("Example 3:")
    id3 = ID3(verbose=False)
    X = [
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
    ]
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    id3.fit(X, Y)

    # prune with medium alpha
    prune(id3.root, X, Y, 5.)

    # show in table
    pred = id3.predict(X)
    table = Table('x', 'y', 'pred')
    for x, y, y_hat in zip(X, Y, pred):
        table.add_row(*map(str, [x, y, y_hat]))
    console.print(table)
