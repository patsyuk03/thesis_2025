import urx
import time
import json
import os
import numpy as np

JSON_PATH_0 = os.path.join(os.path.expanduser("~"), "joint_states_arm_0.json")
with open(JSON_PATH_0, "r") as f:
    recorded_path_0 = list(reversed(json.load(f)))
JSON_PATH_1 = os.path.join(os.path.expanduser("~"), "joint_states_arm_1.json")
with open(JSON_PATH_1, "r") as f:
    recorded_path_1 = list(reversed(json.load(f)))

def get_joint_positions(step):
    joint_pos_arm_0 = recorded_path_0[step]['position']
    joint_pos_arm_1 = recorded_path_1[step]['position']

    vel_arm_0 = recorded_path_0[step]['velocity']
    vel_arm_1 = recorded_path_1[step]['velocity']

    joint_names_arm_0 = recorded_path_0[step]['name']
    joint_names_arm_1 = recorded_path_1[step]['name']

    positions_arm_0 = dict()
    vels_arm_0 = dict()
    for idx, name in enumerate(joint_names_arm_0):
        positions_arm_0[name] = joint_pos_arm_0[idx]
        vels_arm_0[name] = vel_arm_0[idx]

    positions_arm_1 = dict()
    vels_arm_1 = dict()
    for idx, name in enumerate(joint_names_arm_1):
        positions_arm_1[name] = joint_pos_arm_1[idx]
        vels_arm_1[name] = vel_arm_1[idx]

    joint_state_arm_0 = [positions_arm_0['arm_0_shoulder_pan_joint'], positions_arm_0['arm_0_shoulder_lift_joint'], 
                         positions_arm_0['arm_0_elbow_joint'], positions_arm_0['arm_0_wrist_1_joint'],
                         positions_arm_0['arm_0_wrist_2_joint'], positions_arm_0['arm_0_wrist_3_joint']]
    
    joint_state_arm_1 = [positions_arm_1['arm_1_shoulder_pan_joint'], positions_arm_1['arm_1_shoulder_lift_joint'], 
                         positions_arm_1['arm_1_elbow_joint'], positions_arm_1['arm_1_wrist_1_joint'],
                         positions_arm_1['arm_1_wrist_2_joint'], positions_arm_1['arm_1_wrist_3_joint']]
    
    vel_state_arm_0 = [vels_arm_0['arm_0_shoulder_pan_joint'], vels_arm_0['arm_0_shoulder_lift_joint'], 
                       vels_arm_0['arm_0_elbow_joint'], vels_arm_0['arm_0_wrist_1_joint'],
                       vels_arm_0['arm_0_wrist_2_joint'], vels_arm_0['arm_0_wrist_3_joint']]
    
    vel_state_arm_1 = [vels_arm_1['arm_1_shoulder_pan_joint'], vels_arm_1['arm_1_shoulder_lift_joint'], 
                       vels_arm_1['arm_1_elbow_joint'], vels_arm_1['arm_1_wrist_1_joint'],
                       vels_arm_1['arm_1_wrist_2_joint'], vels_arm_1['arm_1_wrist_3_joint']]
    return (joint_state_arm_0, joint_state_arm_1, vel_state_arm_0, vel_state_arm_1)

class RobotControl:
    def __init__(self, ip_0, ip_1):
        self.arm_0 = urx.Robot(ip_0)
        self.arm_1 = urx.Robot(ip_1)
        self.arm_0.secmon.timeout = 2.0
        self.arm_1.secmon.timeout = 2.0
        self.threshold = 0.01
        time.sleep(1)
        print("Connected to UR5 Robot")

    def move_arm_joints(self, joint_state_arm_0, joint_state_arm_1):
        self.arm_0.movej(joint_state_arm_0, acc=0.5, vel=0.5)
        self.arm_1.movej(joint_state_arm_1, acc=0.5, vel=0.5)

    def move_arm_vel(self, joint_state_arm_0, joint_state_arm_1):
        diff_arm_0 = np.abs(np.array(joint_state_arm_0) - np.array(self.get_joints(_print=False)[0]))
        diff_arm_1 = np.abs(np.array(joint_state_arm_1) - np.array(self.get_joints(_print=False)[1]))

        if np.any(diff_arm_0 > self.threshold):
            diff_arm_0 = np.array(joint_state_arm_0) - np.array(self.get_joints(_print=False)[0])
            velocity = diff_arm_0 * 0.5
            # print(velocity)
            self.arm_0.speedj(velocity, acc=1, min_time=0.5)
        if np.any(diff_arm_1 > self.threshold):
            diff_arm_1 = np.array(joint_state_arm_1) - np.array(self.get_joints(_print=False)[1])
            velocity = diff_arm_1 * 0.5
            # print(velocity)
            self.arm_1.speedj(velocity, acc=1, min_time=0.5)

    def get_joints(self, _print=True):
        if _print:
            print("Current joint positions arm_0:", self.arm_0.getj())
            print("Current joint positions arm_1:", self.arm_1.getj())
        return (self.arm_0.getj(), self.arm_1.getj())

    def close_connection(self):
        self.arm_0.close()
        self.arm_1.close()
        print("Disconnected from UR5 Robot")



def main():
    rc = RobotControl("192.168.1.120", "192.168.1.124")
    try:
        joint_state_arm_0, joint_state_arm_1, vel_state_arm_0, vel_state_arm_1 = get_joint_positions(0)
        rc.move_arm_joints(joint_state_arm_0, joint_state_arm_1)
        time.sleep(1)
        print("Moved to initial pose.")

        for step in range(len(recorded_path_0)-2):
            print(f"Step #{step}")
            joint_state_arm_0, joint_state_arm_1, vel_state_arm_0, vel_state_arm_1 = get_joint_positions(step)
            rc.move_arm_vel(joint_state_arm_0, joint_state_arm_1)
            # print(vel_state_arm_0)
            # print(vel_state_arm_1)
            # rc.get_joints()
            # time.sleep(1)

    finally:
        rc.close_connection()

if __name__ == "__main__":
    main()
