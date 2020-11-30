import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

def pageRank(graph, d, max_iteration=1000, epsilon=1e-8):
    """
    given a n * n link graph
    graph[i, j] = 1 means that there is a link from i to j
    d is the proportion of neighbours in the definition of page rank
    return the probablisitic for a user visiting each page
    """
    n, _ = graph.shape
    p = np.ones(n) / n
    graph /= (graph.sum(axis=-1, keepdims=True) + epsilon)
    graph = graph.T
    for i in range(max_iteration):
        pre_p = p
        p = d * graph @ p + (1 - d) / n
        if max(p - pre_p) < epsilon:
            break
    return p

if __name__ == '__main__':
    def demonstrate(graph, d, desc):
        print(desc)
        p = pageRank(graph, d=d)
        print('The probability of each node is', np.round(p, 2))

    graph = np.array(
        [[0, 1, 1, 1],
         [1, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 1, 0]]
    ).astype(float)
    demonstrate(graph, .8, 'Example 1')

    graph = np.array(
        [[0, 1, 1],
         [0, 0, 1],
         [1, 0, 0]]
    ).astype(float)
    demonstrate(graph, .85, 'Example 2')
