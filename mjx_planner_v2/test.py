import numpy as np

a = np.arange(-10, 10).reshape((2,10))
# mask = np.array([1,1,1,0,0], dtype=bool)
b = np.max(a.reshape((2,10, 1)), axis=-1, initial=0)
print(a)
print(b)