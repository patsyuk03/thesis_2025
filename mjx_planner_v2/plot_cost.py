import numpy as np
import matplotlib.pyplot as plt
import os

file_path = f"{os.path.dirname(__file__)}/outputcosts.csv" 

costs = np.genfromtxt(file_path, delimiter=',')

print(costs)

plt.plot(costs)
plt.savefig('costs1.png')
