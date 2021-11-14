import knn_kdtree
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3], [2, 2], [3, 1], [3, 2], [3, 3]])
Y = np.array([0] * len(X))
tree = knn_kdtree.KDTree(X, Y)

def points_equal(a, b):
    a = set(map(tuple, a))
    b = set(map(tuple, b))
    return a == b

assert(points_equal(tree.root.points, [[2, 2]]))
assert(points_equal(tree.root.left.points, [[1, 2]]))
assert(points_equal(tree.root.right.points, [[3, 2]]))
assert(points_equal(tree.root.left.left.points, [[1, 1]]))
assert(points_equal(tree.root.left.right.points, [[1, 3]]))
assert(points_equal(tree.root.right.left.points, [[3, 1]]))
assert(points_equal(tree.root.right.right.points, [[3, 3]]))

assert(points_equal([a[0] for a in tree.query(np.array([2, 1]), 3)], [[1, 1], [2, 2], [3, 1]]))

X = np.array([[0, 0], [1, 1], [2, 2]])
Y = np.array([0] * len(X))
tree = knn_kdtree.KDTree(X, Y)
assert(points_equal([a[0] for a in tree.query(np.array([1, 1]), 3)], X))

X = np.array([[0, 0], [1, 1], [2, 2]])
Y = np.array([0] * len(X))
tree = knn_kdtree.KDTree(X, Y)
assert(points_equal([a[0] for a in tree.query(np.array([10, 2.001]), 3)], X))
