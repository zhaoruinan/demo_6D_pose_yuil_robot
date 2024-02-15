from yuil_lib import Yuil_robot
from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
import time
def test1():
    real_robot = Yuil_robot()
    xyz = np.array([[0.601], [-0.323], [0.377]])
    abc = np.array([-3.14, 0.0, 1.571])
    #real_robot.go_home()
    real_robot.xyz_to_joint_move(xyz,abc,10)
    time.sleep(15)


    real_robot.gripper_close()
    time.sleep(2)
    real_robot.gripper_open()
    xyz = np.array([[0.801], [-0.123], [0.177]])
    abc = np.array([-3.14, 0.0, -3.142])
    time.sleep(2)

    real_robot.xyz_to_joint_move(xyz,abc,10)

def test2():
    real_robot = Yuil_robot()
    xyz = np.array([[0.601], [-0.323], [0.377]])
    abc = np.array([-3.14, 0.0, 1.571])
    real_robot.go_home(200)
    time.sleep(10)


    real_robot.gripper_close()
    time.sleep(2)
    real_robot.gripper_open()
    xyz = np.array([[0.851], [-0.123], [0.077]])
    abc = np.array([-3.14, 0.0, -3.142])
    time.sleep(2)

    real_robot.xyz_to_joint_move(xyz,abc,100)
def test3():
    real_robot = Yuil_robot()
    real_robot.go_home(80)

    time.sleep(10)


    xyz = np.array([[0.651], [-0.123], [0.177]])
    abc = np.array([-3.14, 0.0, -3.142])
    pos = [0.651, -0.123, 0.077,-3.14, 0.0, 1.56]
    #real_robot.xyz_to_joint_move(xyz,abc,10)
    real_robot.xyz_move(pos,80)

def main():
    test3()

if __name__ == "__main__":
    main()