import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading

class ForceTorquePlotter(Node):

    def __init__(self):
        super().__init__('force_torque_plotter')
        
        # Subscriber to the /wrench topic to get force/torque data
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.listener_callback,
            10
        )
        
        self.force_data = []
        self.torque_data = []
        self.time_stamps = []

        # Plot setup
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6))
        self.ax1.set_title('Force (X, Y, Z)')
        self.ax1.set_ylabel('Force (N)')
        self.ax1.set_ylim([-10, 10])
        self.ax2.set_title('Torque (X, Y, Z)')
        self.ax2.set_ylabel('Torque (Nm)')
        self.ax2.set_ylim([-10, 10])
        self.ax2.set_xlabel('Time Steps')

        # Initialize lines
        self.force_lines = [self.ax1.plot([], [], label=f'Force {axis}')[0] for axis in 'XYZ']
        self.torque_lines = [self.ax2.plot([], [], label=f'Torque {axis}')[0] for axis in 'XYZ']
        self.ax1.legend()
        self.ax2.legend()



    def listener_callback(self, msg):
        # Append new data from the wrench message
        force = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        torque = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        self.force_data.append(force)
        self.torque_data.append(torque)
        self.time_stamps.append(len(self.time_stamps))
        self.update_plot()


    def update_plot(self):
        # Update plot lines for force and torque
        for i, line in enumerate(self.force_lines):
            line.set_data(self.time_stamps, [f[i] for f in self.force_data])
        for i, line in enumerate(self.torque_lines):
            line.set_data(self.time_stamps, [t[i] for t in self.torque_data])

        # Set plot limits
        self.ax1.set_xlim([max(0, len(self.time_stamps) - 100), len(self.time_stamps)])
        self.ax2.set_xlim([max(0, len(self.time_stamps) - 100), len(self.time_stamps)])

        # Draw the updated plot
        self.ax1.relim()
        self.ax2.relim()
        self.ax1.autoscale_view(True, True, True)
        self.ax2.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

def main(args=None):
    rclpy.init(args=args)
    node = ForceTorquePlotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
