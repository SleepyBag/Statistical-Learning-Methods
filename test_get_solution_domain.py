from utils import row_echelon, get_solution_domain
import numpy as np

for i in range(100):
    print('processing ', i)
    a = np.random.rand(100, 50)
    re = row_echelon(a)
    assert(len(re) == np.linalg.matrix_rank(a))
    zero = a @ get_solution_domain(re)
    assert((zero == 0.).all())

for i in range(100):
    print('processing ', i)
    a = np.random.rand(5, 10)
    re = row_echelon(a)
    assert(len(re) == np.linalg.matrix_rank(a))
    zero = a @ get_solution_domain(re)
    assert((abs(zero) < 1e-8).all())
