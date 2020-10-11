from utils import line_search

class F:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return (x - self.n) ** 2

f = F(0)
epsilon = 1e-6
for i in range(-1000, 1000):
    f.n = i
    assert(abs(line_search(f, -2000, 2000, epsilon) - i) <= epsilon)
