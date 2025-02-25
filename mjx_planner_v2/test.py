import numpy as np

# a  = np.array([
#     [11,12,13,14,21,22,23,24],
#     [11,12,13,14,21,22,23,24],
#     [11,12,13,14,21,22,23,24],
#     [11,12,13,14,21,22,23,24],
#     [11,12,13,14,21,22,23,24],
# ])
# # print(a[0].reshape(2,4))
# batch, num_dof, num_steps = 5, 2, 4
# print(batch, num_dof, num_steps)
# print(a.reshape(batch*num_dof, num_steps).T)
# b = a.reshape(batch*num_dof, num_steps).T
# print(b[0].reshape(batch, num_dof))
# out = b[0].reshape(batch, num_dof).flatten()
# print(b.T.reshape(batch, num_dof*num_steps))
# print(np.max([True, False, True]))
a = [1,2,3,4,5]
b = [1,2,3, 9]
d = [c in a for c in b]
# print(np.max(d))
# print(np.isin(b, a))
print(np.where(True, 100, 0))

# print(out)
# print(a)