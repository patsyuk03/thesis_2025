import numpy as np
import os


file_path = f"{os.path.dirname(__file__)}/collision.csv" 
collision = np.genfromtxt(file_path, delimiter=',')

print(collision.shape)
print(collision[0])