from utils import Heap

heap = Heap([3, 1, 2])
heap.push(1)
heap.push(2)
a = [i for i in heap]

assert(a == [1, 1, 2, 2, 3])
