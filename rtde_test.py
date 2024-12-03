from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Connect to robot
rtde_c = RTDEControl("192.168.1.124")
rtde_r = RTDEReceive("192.168.1.124")

# Get current joint positions
joint_positions = rtde_r.getActualQ()
print("Current joint positions:", joint_positions)

new_position = [-2.210294548665182, -1.5044668477824708, 1.7937071959124964, -2.021080633203024, -1.5705907980548304, -3.1616676489459437]
rtde_c.moveJ(new_position)

# Disconnect
rtde_c.disconnect()
