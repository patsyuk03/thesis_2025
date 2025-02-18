import numpy as np
import matplotlib.pyplot as plt
import os

# file_path = f"{os.path.dirname(__file__)}/cost_g_best.csv" 
file_path = f"{os.path.dirname(__file__)}/output_costs1.csv" 


costs = np.genfromtxt(file_path, delimiter=',')

print(costs)

plt.plot(costs)
plt.savefig('costs.png')
