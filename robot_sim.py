from yuil_lib import Yuil_robot
from visual_kinematics.RobotSerial import *
import numpy as np
from math import cos,sin,pi
import time
import cv2
from cv2 import aruco
import glob
import matplotlib.pyplot as plt
import os, shutil
from scipy.spatial.transform import Rotation


class robot_sim(object):
    def __init__(self):
        dh_params = np.array([[0.138, 0.,  0.5*pi, 0],
                              [0., 0.42135, 0., 0.5 * pi],
                              [0., 0.40315, 0., -0.5 * pi],
                              [0.123, 0., 0.5 * pi, 0.5 * pi],
                              [0.098, 0., -0.5 * pi, 0.5 * pi],
                              [0.082, 0.,  0., 0.5*pi]])

        self.serial = RobotSerial(dh_params)