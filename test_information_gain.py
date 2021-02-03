from utils import information_gain, entropy
from collections import Counter
from math import fabs

eps = 1e-3

X = [
    ['青年', '否', '否', '一般'],
    ['青年', '否', '否', '好'],
    ['青年', '是', '否', '好'],
    ['青年', '是', '是', '一般'],
    ['青年', '否', '否', '一般'],
    ['中年', '否', '否', '一般'],
    ['中年', '否', '否', '好'],
    ['中年', '是', '是', '好'],
    ['中年', '否', '是', '非常好'],
    ['中年', '否', '是', '非常好'],
    ['老年', '否', '是', '非常好'],
    ['老年', '否', '是', '好'],
    ['老年', '是', '否', '好'],
    ['老年', '是', '否', '非常好'],
    ['老年', '否', '否', '一般'],
]
Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']

assert(fabs(entropy(Counter(Y).values()) - .971) < eps)
assert(fabs(information_gain(X, Y, 0) - .083) < eps)
assert(fabs(information_gain(X, Y, 1) - .324) < eps)
assert(fabs(information_gain(X, Y, 2) - .420) < eps)
assert(fabs(information_gain(X, Y, 3) - .363) < eps)
