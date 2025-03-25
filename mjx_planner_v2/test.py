import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create the figure and axis
fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], 'r-')

# Initialize the plot
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

def update(frame):
    xdata.append(frame / 10)
    ydata.append(np.sin(frame / 10 * 2 * np.pi))
    line.set_data(xdata, ydata)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(100), blit=True)

plt.show()
