import numpy as np
import matplotlib.pyplot as plt
import os

output_cost_file_path = f"{os.path.dirname(__file__)}/data/output_costs.csv" 
best_cost_g_file_path = f"{os.path.dirname(__file__)}/data/best_cost_g.csv" 
best_vels_file_path = f"{os.path.dirname(__file__)}/data/best_vels.csv" 
best_traj_file_path = f"{os.path.dirname(__file__)}/data/best_traj.csv" 

output_cost = np.genfromtxt(output_cost_file_path, delimiter=',')
best_cost_g = np.genfromtxt(best_cost_g_file_path, delimiter=',')
best_vels = np.genfromtxt(best_vels_file_path, delimiter=',')
best_traj = np.genfromtxt(best_traj_file_path, delimiter=',')

plt.plot(output_cost)
plt.title("Output Costs")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.savefig(f'{os.path.dirname(__file__)}/plots/output_cost.png')
plt.clf()

plt.plot(best_cost_g)
plt.title("Best Cost Goal")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.savefig(f'{os.path.dirname(__file__)}/plots/best_cost_g.png')
plt.clf()

plt.plot(best_vels)
plt.title("Best Velocities")
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
plt.savefig(f'{os.path.dirname(__file__)}/plots/best_vels.png')
plt.clf()

plt.plot(best_traj)
plt.title("Best Trajectory")
plt.xlabel("Step")
plt.ylabel("Joint States")
plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
plt.savefig(f'{os.path.dirname(__file__)}/plots/best_traj.png')
plt.clf()
