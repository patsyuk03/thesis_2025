import matplotlib.pyplot as plt
import os


file_path = f"{os.path.dirname(__file__)}/costs1.txt" 
with open(file_path, 'r') as file:
    lines = file.readlines()

costs = [float(line.rstrip()) for line in lines]

print(costs)

plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


