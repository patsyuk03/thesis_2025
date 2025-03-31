import numpy as np
import matplotlib.pyplot as plt
import os

costs_file_path = f"{os.path.dirname(__file__)}/data/costs.csv" 
cost_g_file_path = f"{os.path.dirname(__file__)}/data/cost_g.csv" 
cost_r_file_path = f"{os.path.dirname(__file__)}/data/cost_r.csv" 
cost_c_file_path = f"{os.path.dirname(__file__)}/data/cost_c.csv" 
thetadot_file_path = f"{os.path.dirname(__file__)}/data/thetadot.csv" 
theta_file_path = f"{os.path.dirname(__file__)}/data/theta.csv" 
# history_file_path = f"{os.path.dirname(__file__)}/data/history.csv" 


costs = np.genfromtxt(costs_file_path, delimiter=',')
cost_g = np.genfromtxt(cost_g_file_path, delimiter=',')
cost_r = np.genfromtxt(cost_r_file_path, delimiter=',')
cost_c = np.genfromtxt(cost_c_file_path, delimiter=',')

thetadot = np.genfromtxt(thetadot_file_path, delimiter=',')
theta = np.genfromtxt(theta_file_path, delimiter=',')
# history = np.genfromtxt(history_file_path, delimiter=',')


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




plt.plot(costs)
plt.title("Costs")
plt.xlabel("Iteration")
plt.ylabel("Cost")
# plt.show()
plt.savefig(f'{os.path.dirname(__file__)}/plots/costs.png')
plt.clf()

plt.plot(cost_g)
plt.title("Cost Goal")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.savefig(f'{os.path.dirname(__file__)}/plots/cost_g.png')
plt.clf()

plt.plot(cost_r)
plt.title("Cost Rotation")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.savefig(f'{os.path.dirname(__file__)}/plots/cost_r.png')
plt.clf()

plt.plot(cost_c)
plt.title("Cost Collision")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.savefig(f'{os.path.dirname(__file__)}/plots/cost_c.png')
plt.clf()

plt.plot(thetadot)
plt.title("Velocities")
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
plt.savefig(f'{os.path.dirname(__file__)}/plots/thetadot.png')
plt.clf()

plt.plot(theta)
plt.title("Trajectory")
plt.xlabel("Step")
plt.ylabel("Joint States")
plt.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], loc='upper left')
plt.savefig(f'{os.path.dirname(__file__)}/plots/theta.png')
plt.clf()
