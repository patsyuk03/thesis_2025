import numpy as np

a = np.arange(3)
a = np.tile(a, (10, 1))

b = np.array([4, 5, 6])
b = np.tile(b, (10, 1))


c = np.linalg.norm(a - b, axis = 1)

print(c)
