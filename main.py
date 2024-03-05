from yuil_lib import Yuil_robot
from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
import time

class robot_sim(object):
    def __init__(self):
        self.dh_params = np.array([[0.138, 0.,  0.5*pi, 0],
                              [0., 0.42135, pi, 0.5 * pi],
                              [0., 0.40315, pi, 0.],
                              [0.123, 0., -0.5 * pi, -0.5 * pi],
                              [0.098, 0., 0.5 * pi, 0.],
                              [0.082, 0.,  0., 0.]])

        #self.serial = RobotSerial(dh_params)
    def t_4_4(self,joint):
        joint = joint * pi/180
        f = self.serial.forward(joint)
        print(f.t_3_1.reshape([3,]),f.euler_3)
        return f.t_4_4
    def abc_euler(self,joint):
        dh_params = np.array([[0.138, 0.,  0.5*pi, 0],
                              [0., 0.42135, 0., 0.5 * pi],
                              [0., 0.40315, 0., -0.5 * pi],
                              [0.123, 0., 0.5 * pi, 0.5 * pi],
                              [0.098, 0., -0.5 * pi, 0.5 * pi],
                              [0.082, 0.,  0., 0.5*pi]])
        serial = RobotSerial(dh_params)    
        joint = joint * pi/180
        f = serial.forward(joint)
        return f.t_3_1.reshape([3, ]), f.euler_3
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
    real_robot.go_home(100)

    time.sleep(10)


    xyz = np.array([[0.651], [-0.123], [0.177]])
    abc = np.array([-3.14, 0.0, -3.142])
    pos = [0.551, 0.383, 0.277,-3.14, 0.0, 1.56]
    #real_robot.xyz_to_joint_move(xyz,abc,10)
    real_robot.xyz_move(pos,90)
def test4():
    real_robot = Yuil_robot()
    sim = robot_sim()
    real_robot.gripper_open()

    time.sleep(2)
    real_robot.go_home(95)
    time.sleep(10)

    real_robot.gripper_close()
    dh_params = np.array([[0.138, 0.,  0.5*pi, 0],
                              [0., 0.42135, 0., 0.5 * pi],
                              [0., 0.40315, 0., -0.5 * pi],
                              [0.123, 0., 0.5 * pi, 0.5 * pi],
                              [0.098, 0., -0.5 * pi, 0.5 * pi],
                              [0.082, 0.,  0., 0.5*pi]]) 
    serial = RobotSerial(dh_params)    
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 8089))
    
    print("server init")
    s.listen(1)
    while(True):
        conn, addr = s.accept()
        cmnd = conn.recv(4)  # The default size of the command packet is 4 bytes
        print(cmnd)
        if 'READ' in str(cmnd):
            # Do the initialization action
            pos_j = real_robot.robot_get_current_position()
            pos_j = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            print(pos_j[0])
            joint = pos_j * pi/180
            f = serial.forward(joint)
            abc,euler = f.t_3_1.reshape([3, ]), f.euler_3
            pos_t = np.array([abc[0],abc[1],abc[2],euler[0],euler[1],euler[2]])
            
            pos_j = real_robot.robot_get_current_xyz_position()
            pos_t = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            print(pos_t)
            conn.sendall(pos_t)

        elif 'HOME' in str(cmnd):
            # Do the play action
            real_robot.go_home(95)
            #time.sleep(10)
            pos = [0.551, 0.383, 0.277,-3.14, 0.0, 1.56]
            #real_robot.xyz_move(pos,90)

            pos_j = real_robot.robot_get_current_position()
            pos_j = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            print(type(pos_j[0]))
            conn.sendall(pos_j)
            #pose = real_robot.xyz_move(pos,80)
        elif 'PS_T' in str(cmnd):
            # Do the play action
            pose = conn.recv(1024)
            pose_set = np.frombuffer(pose, dtype=np.float64)
            print(f"Received {pose!r}")
            print(pose_set)
            real_robot.xyz_move(pose_set,99)

            pos_j = real_robot.robot_get_current_position()
            pos_j = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            conn.sendall(pos_j)
            #pose = real_robot.xyz_move(pos,80)
        elif 'PS_J' in str(cmnd):
            # Do the play action
            pose = conn.recv(1024)
            pose_set = np.frombuffer(pose, dtype=np.float64)
            print(f"Received {pose!r}")
            print(pose_set)
            real_robot.robot_movj(pose_set,90)

            pos_j = real_robot.robot_get_current_position()
            pos_j = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            conn.sendall(pos_j)
            #pose = real_robot.xyz_move(pos,80)
        elif 'QUIT' in str(cmnd):
            # Do the quiting action
            conn.sendall(b'QUIT-DONE')
            break
        else:
            conn.sendall(b'not a cmd')
    s.close()
def main():
    test4()

if __name__ == "__main__":
    main()