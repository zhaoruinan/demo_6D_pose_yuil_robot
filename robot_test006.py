import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import sys
import imagezmq
import codecs, json 
import math
import time
import gripper
import threading
from pynput import keyboard
from xarm.wrapper import XArmAPI
from configparser import ConfigParser
pi = math.pi
from math import cos,sin
class Task:
    def __init__(self,robot):
        self.task_list=[]
        self.task_flag="wait"
        self._robot=robot
        self._info={"robot_goal":[6.699997, -24.80002, -26.199985, -0.50002, 47.799978, -5],"grasp_angle":0.0,"joint_p":self._robot.get_angle(),"tool_p":self._robot.get_position()}
    def add(self,goal):
        self.task_list.append(goal)
    def do_task(self,task_name):
        if task_name=="h":   # go home postion
            self._robot.go_home()
        if task_name=="o": # gripper_open
            gripper.open()
        if task_name=="c":#gripper_close
            gripper.close()
        if task_name=="do":
            servo_angle = self._robot.get_inverse_kinematics(self._info['robot_goal'])
            self._robot.go_action(servo_angle)
            gripper.close()
        if task_name=="u":
            self._info["joint_p"]=self._robot.get_angle()
            self._info["tool_p"]=self._robot.get_position()
        self.change_flag("wait")


#        return self.task_list.pop()
    def flag(self):
        return self.flag
    def change_flag(self,input):
        self.flag=input
    def info_update(self,info_name,info):
        try:
            self._info[info_name]=info
        except:
            print("No this info")
    def task_input(self):
        with keyboard.Events() as events:        
            event = events.get(1.0)
        if event is None:
            time.sleep(0.5)
            #print('You did not press a key within one second')
        else:
            print('Received event {}'.format(event))
            self.change_flag("wait_cmd")
            print("robot_goal",self._info['robot_goal'])
            print("grasp_angle",self._info['grasp_angle'])
            print("tool_p",self._info['tool_p'])
            print("joint_p",self._info['joint_p'])
            cmd = input("Please input CMD\n")
            self.do_task(cmd)

    
def main():
    robot = Robot()
    task = Task(robot)
    task.do_task("h")
    task.do_task("o")
    p_camera = threading.Thread(target=camera,args=(task,))
    p_camera.start()
    while True:
        task.task_input()
#    arm.go_home()
#    time.sleep(5)
#    gripper.open()
    

    
    # go action
    #print(arm.get_position(), arm.get_position())
    #go_check = input("Please check and input ok\n")
    #go_check = 'ok'
    #time.sleep(5)
    #gripper.set(0)
    #print(arm.get_position())

def camera(task):
    sender = imagezmq.ImageSender(connect_to='tcp://192.168.10.2:5555')
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display

        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        text1 = sender.send_image('RealSense', color_image)
        json_load = json.loads(text1)
        corner_2d_pred=np.asarray(json_load['corner_2d_pred'])
        pose_pred = np.asarray(json_load['pose_pred'])

        points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
        points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
        #demo_image_ = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)
        demo_image_ = np.uint8(color_image).copy()
        

        try:
            cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
            cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
        except:
            pass
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, demo_image_))
        else:
            images = np.hstack((color_image, demo_image_))
        #cv2.imshow('RealSense', images)
        #cv2.waitKey(1)

        # check grasp postion
        task.do_task("u")
        xyz1 = task._info["tool_p"]
        #print("radian",xyz1[-1])
        t2b_R,t2b_T = gripper2base(xyz1[-1][3],xyz1[-1][4],xyz1[-1][5],xyz1[-1][0]/1000,xyz1[-1][1]/1000,xyz1[-1][2]/1000)
        #print('t2b',t2b_R,t2b_T)
        t2b_TR = RpToTrans(t2b_R,t2b_T)
        #print("t2b_TR",t2b_TR)
        c2t_TR = np.matrix([[ 0.74801657, -0.66357547, -0.01178181, -0.04541425],
                       [ 0.66367645,  0.7479531,   0.00998636,  0.0234043 ],
                       [ 0.00218554, -0.01528927,  0.99988072, -0.05862435],
                       [ 0.,          0.,          0.,          1.        ]])
        c2b = np.dot(t2b_TR, c2t_TR)
        #print('c2b',c2b)
        #print("11111111\n")
        #print(pose_pred[:, :3].T)
        #print("22222222\n")
        #print(pose_pred[:, 3:].T)    

        o2c_TR = RpToTrans(pose_pred[:, :3].T,pose_pred[:, 3:].T[0])
        #print("o2c_TR",o2c_TR)
        o2b_TR = np.dot(t2b_TR,o2c_TR)
        o2b_R,o2b_T = TransToRp(o2b_TR) 
        #print("o2b_TR",o2b_TR)
        #print("o2b_R",o2b_R)
        #print("o2b_T",o2b_T)
        move_goal_R = rotm2euler(o2b_R)
        d = 0.0 #6
        #if move_goal_R[2] < 0:
        #    move_goal_R[2] = move_goal_R[2] +pi
        if move_goal_R[2] > 0:
            move_goal_R[2] = move_goal_R[2] -pi
        d_b = move_goal_R[2] - xyz1[1][5]#+pi/6
        move_fix = [d*sin(d_b),-d*cos(d_b),0.12]
        move_goal_T = o2b_T + move_fix
        #print(move_goal_T)
        x1 = pose_pred[:, 3:].T[0][1]
        #print(x1)
        y1 = pose_pred[:, 3:].T[0][0]
        #print(y1)
        #print(xyz1)
        move_goal_T[0] = xyz1[1][0]/1000-x1+move_fix[0] -0.07
        move_goal_T[1] = xyz1[1][1]/1000-y1+move_fix[1] +0.06
        move_goal_T[2] = xyz1[1][2]/1000-move_fix[2]
        #move_goal = [move_goal_T[0]*1000,move_goal_T[1]*1000,move_goal_T[2]*1000,,0,0]
        move_goal = [move_goal_T[0]*1000,move_goal_T[1]*1000,move_goal_T[2]*1000,pi,0,xyz1[1][5]]#move_goal_R[2]]
        #print(pose_pred[:, 3:].T[0])
        if task.flag =="wait":
            #if move_goal_T[2]>0.012 and move_goal_T[2]<0.2 :
                task.info_update("robot_goal",move_goal)
                task.info_update("grasp_angle",d_b*180/pi)




        # Stop streaming
    pipeline.stop()


class Robot:
    def __init__(self):
        parser = ConfigParser()
        parser.read('./robot.conf')
        ip = parser.get('xArm', 'ip')
        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
    def go_home(self):
        speed = math.radians(800)
        self.arm.set_servo_angle(angle=[6.699997, -24.80002, -26.6, -0.5, 51.4, 59.2], speed=speed,is_radian=False, wait=True)
    def go_action(self,servo_angle):
        speed = math.radians(800)
        #self.arm.set_servo_angle(angle=[6.699997, -24.80002, -26.199985, -0.50002, 50, -16], speed=speed, is_radian=False,wait=True)
        #print("input servo_angle",servo_angle)
        self.arm.set_servo_angle(angle=servo_angle[1],speed=speed, is_radian=False,wait=True)
    def get_position(self):
        return self.arm.get_position(is_radian=True)
    def get_angle(self):
        return self.arm.get_servo_angle(is_radian=False)
    def get_inverse_kinematics(self,pose):
        return self.arm.get_inverse_kinematics(pose,input_is_radian=True, return_is_radian=False)

def angle2rotation(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R
def gripper2base(x, y, z, tx, ty, tz):
    thetaX = x #/ 180 * pi
    thetaY = y #/ 180 * pi
    thetaZ = z #/ 180 * pi
    R_gripper2base = angle2rotation(thetaX, thetaY, thetaZ)
    T_gripper2base = np.array([[tx], [ty], [tz]])
    Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
    Matrix_gripper2base = np.row_stack((Matrix_gripper2base, np.array([0, 0, 0, 1])))
    R_gripper2base = Matrix_gripper2base[:3, :3]
    T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
    return R_gripper2base, T_gripper2base

def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if ((abs(R[0][1]-R[1][0])< epsilon) and (abs(R[0][2]-R[2][0])< epsilon) and (abs(R[1][2]-R[2][1])< epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1]+R[1][0]) < epsilon2) and (abs(R[0][2]+R[2][0]) < epsilon2) and (abs(R[1][2]+R[2][1]) < epsilon2) and (abs(R[0][0]+R[1][1]+R[2][2]-3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0,1,0,0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0]+1)/2
        yy = (R[1][1]+1)/2
        zz = (R[2][2]+1)/2
        xy = (R[0][1]+R[1][0])/4
        xz = (R[0][2]+R[2][0])/4
        yz = (R[1][2]+R[2][1])/4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx< epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy< epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz< epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z
        return [angle,x,y,z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2])*(R[2][1] - R[1][2]) + (R[0][2] - R[2][0])*(R[0][2] - R[2][0]) + (R[1][0] - R[0][1])*(R[1][0] - R[0][1])) # used to normalise
    if (abs(s) < 0.001):
        s = 1 

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1)/2)
    x = (R[2][1] - R[1][2])/s
    y = (R[0][2] - R[2][0])/s
    z = (R[1][0] - R[0][1])/s
    return [angle,x,y,z]
def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def TransInv(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector
    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat
    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]
#if __name__ =='__main__':
main()