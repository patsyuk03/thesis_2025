import numpy as np
import cv2

arr= np.arange(24)
mat = arr.reshape((2,3,4))
print(mat)

print(mat.reshape((mat.shape[0]*mat.shape[1], mat.shape[2])))