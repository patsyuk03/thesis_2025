import numpy as np
import matplotlib.pyplot as plt
import os

output_cost_file_path = f"{os.path.dirname(__file__)}/data/output_costs.csv" 
best_cost_g_file_path = f"{os.path.dirname(__file__)}/data/best_cost_g.csv" 
best_vels_file_path = f"{os.path.dirname(__file__)}/data/best_vels.csv" 
best_traj_file_path = f"{os.path.dirname(__file__)}/data/best_traj.csv" 
history_file_path = f"{os.path.dirname(__file__)}/data/history.csv" 


output_cost = np.genfromtxt(output_cost_file_path, delimiter=',')
best_cost_g = np.genfromtxt(best_cost_g_file_path, delimiter=',')
best_vels = np.genfromtxt(best_vels_file_path, delimiter=',')
best_traj = np.genfromtxt(best_traj_file_path, delimiter=',')
history = np.genfromtxt(history_file_path, delimiter=',')


# num_batches = np.arange(100, 1000, 100)
# plt.plot(num_batches, history)
# plt.title("Time History")
# plt.xlabel("Batch Size")
# plt.ylabel("Time (s)")
# plt.savefig(f'{os.path.dirname(__file__)}/plots/history.png')
# plt.clf()

# fig, axs = plt.subplots(3, figsize=(5, 10))
# fig.suptitle('History (30 iter)')
# axs[0].plot(history[-1], history[0])
# axs[0].set_title('Step Time')
# axs[0].set(xlabel='Batch Size', ylabel='Time (s)')
# axs[1].plot(history[-1], history[1])
# axs[1].set_title('Simulation Time')
# axs[1].set(xlabel='Batch Size', ylabel='Time (s)')
# axs[2].plot(history[-1], history[2])
# axs[2].set_title('Cost')
# axs[2].set(xlabel='Batch Size', ylabel='Cost')
# fig.tight_layout()
# plt.savefig(f'{os.path.dirname(__file__)}/plots/history.png')
# plt.clf()




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

print(best_vels[0])
plt.plot(best_vels[:20])
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
