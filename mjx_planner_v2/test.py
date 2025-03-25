import numpy as np

def quaternion_distance(q1, q2):

    dot_product = np.abs(np.dot(q1/np.linalg.norm(q1, axis=1).reshape(1, 5).T, q2/np.linalg.norm(q2)))
    print(q1.shape, np.linalg.norm(q1, axis=1).T.shape)
    return 2 * np.arccos(dot_product)



q1 = np.tile(np.array([1, 0, 1, 0]), (5, 1))
# q1 = np.array([1, 0, 1, 0], dtype=float)
q2 = np.array([1, 0, 1, 0], dtype=float)

# print(q1)


a = quaternion_distance(q1, q2)
print(a)