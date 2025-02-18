import numpy as np
import matplotlib.pyplot as plt
import os

file_path = f"{os.path.dirname(__file__)}/best_vels.csv" 

thetadot = np.genfromtxt(file_path, delimiter=',')

print(thetadot.shape)

for i in range(thetadot.shape[0]):
    theta_single = thetadot[i].T
    plt.plot(theta_single)
plt.show()