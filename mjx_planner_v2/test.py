import numpy as np
import os


# file_path = f"{os.path.dirname(__file__)}/collision.csv" 
# collision = np.genfromtxt(file_path, delimiter=',')

# print(collision.shape)
# print(collision[0])

a = np.array([1,1,1])
b = np.array([2,2,2])

c = np.concatenate((a,b), axis=0)
c = c.reshape((2, c.shape[0]//2))

print(c)