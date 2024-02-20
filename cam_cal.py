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

CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
CHARUCO_BOARD = aruco.CharucoBoard_create(squaresX=CHARUCOBOARD_COLCOUNT, squaresY=CHARUCOBOARD_ROWCOUNT, squareLength=0.04, markerLength=0.02, dictionary=ARUCO_DICT)

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
                    [0,         cos(theta[0]), -sin(theta[0]) ],
                    [0,         sin(theta[0]), cos(theta[0])  ]
                    ])
    R_y = np.array([[cos(theta[1]),    0,      sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-sin(theta[1]),   0,      cos(theta[1])  ]
                    ])         
    R_z = np.array([[cos(theta[2]),    -sin(theta[2]),    0],
                    [sin(theta[2]),    cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

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

def test3():
    real_robot = Yuil_robot()
    #real_robot.go_home(80)
    vid = cv2.VideoCapture(0) 
    start_time = time.time()
    run_num = 0.0
    pos = [0.651, -0.123, 0.277,-3.14, 0.0, 1.56]
    while(True): 
        ret, frame = vid.read() 
        if ret==True:
            cv2.imshow('RGB', frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break
        
        if (time.time() - start_time)%5 < 0.05 and run_num < (time.time() - start_time)/5:
            cv2.imwrite("cal_data/002/RGBimgs/color_image" + str(int(run_num)+1) + ".png", frame)
            print((time.time() - start_time))
            pos_n = real_robot.robot_get_current_xyz_position()
            print(pos_n[0],pos_n[1],pos_n[2],pos_n[3],pos_n[4],pos_n[5])
            pos_j = real_robot.robot_get_current_position()
            print(pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5])
            run_num = run_num +1
            if run_num%2 == 0:
                pos[0]= pos[0]+0.05
            else:
                pos[0]=0.651
            real_robot.xyz_move(pos,80)

    vid.release() 
    cv2.destroyAllWindows() 

class CameraCalibration:
    """Camera calibration class, this class takes as input a folder with images and a folder with the corresponding Base2endeffector transforms
    and outputs the intrinsic matrix in a .npz file. It also performs hand-eye calibration and saves those results in a .npz file.
    The images with the corner detection are saved in a folder called 'DetectedCorners'

    This class has 4 optional parameters:
    pattern_size: the number of corners in the chessboard pattern, default is (4,7)
    square_size: the size of the squares in the chessboard pattern, default is 33/1000
    ShowProjectError: if True, it will show the reprojection error for each image in a bar plot, default is False
    ShowCorners: if True, it will show the chessboard corners for each image, default is False

    """
    def __init__(self, image_folder, Transforms_folder, pattern_size=(4, 7), square_size=33/1000, ShowProjectError=False, ShowCorners=False, charuco_use=False):

        #Initiate parameters
        self.pattern_size = pattern_size
        self.square_size = square_size

        #load images and joint positions
        self.image_files = sorted(glob.glob(f'{image_folder}/*.png'))
        self.transform_files = sorted(glob.glob(f'{Transforms_folder}/*.npz'))
        self.images = [cv2.imread(f) for f in self.image_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in self.images]
        self.All_T_base2EE_list = [np.load(f)['arr_0'] for f in self.transform_files]

        #find chessboard corners and index of images with chessboard corners
        self.chessboard_corners, self.IndexWithImg, self.ids_all = self.find_chessboard_corners(self.images, self.pattern_size, ShowCorners=ShowCorners,charuco_use=charuco_use)
        if charuco_use:
            self.intrinsic_matrix = self.calculate_intrinsics_charuco(self.chessboard_corners, self.ids_all,
                                                           self.pattern_size, self.square_size,
                                                           self.images[0].shape[:2], ShowProjectError = ShowProjectError)
        else:
            self.intrinsic_matrix = self.calculate_intrinsics(self.chessboard_corners, self.IndexWithImg,
                                                           self.pattern_size, self.square_size,
                                                           self.images[0].shape[:2], ShowProjectError = ShowProjectError)
        print("intrinsic_matrix",self.intrinsic_matrix)

        #Remove transforms were corners weren't detected
        #print(self.All_T_base2EE_list)
        #print(self.IndexWithImg)
        self.T_base2EE_list = [self.All_T_base2EE_list[i] for i in self.IndexWithImg]
        print("T_base2EE",self.T_base2EE_list[0])

        #save intrinsic matrix
        np.savez("IntrinsicMatrix.npz", self.intrinsic_matrix)
        #Calculate camera extrinsics
        self.RTarget2Cam, self.TTarget2Cam = self.compute_camera_poses(self.chessboard_corners,
                                                                       self.pattern_size, self.square_size,
                                                                       self.intrinsic_matrix,charuco_use=charuco_use ,ids_all=self.ids_all)

        #Convert to homogeneous transformation matrix
        self.T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(self.RTarget2Cam, self.TTarget2Cam)]
        for i in range(len(self.T_target2cam)):
            self.T_target2cam[i] = np.concatenate((self.T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)

        #Calculate T_cam2target
        self.T_cam2target = [np.linalg.inv(T) for T in self.T_target2cam]
        self.R_cam2target = [T[:3, :3] for T in self.T_cam2target]
        self.R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in self.R_cam2target]
        self.T_cam2target = [T[:3, 3] for T in self.T_cam2target]   #4x4 transformation matrix

        #Calculate T_Base2EE

        self.TEE2Base = [np.linalg.inv(T) for T in self.T_base2EE_list]
        self.REE2Base = [T[:3, :3] for T in self.TEE2Base]
        self.R_vecEE2Base = [cv2.Rodrigues(R)[0] for R in self.REE2Base]
        self.tEE2Base = [T[:3, 3] for T in self.TEE2Base]
       #Create folder to save final transforms
        if not os.path.exists("cal_data/002/FinalTransforms/"):
            os.mkdir("cal_data/002/FinalTransforms/")
        #solve hand-eye calibration
        for i in range(0, 5):
            print("Method:", i)
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(
                self.R_cam2target,
                self.T_cam2target,
                self.R_vecEE2Base,
                self.tEE2Base,
                method=i
            )

            #print and save each results as .npz file
            print("The results for method", i, "are:")
            print("R_cam2gripper:", self.R_cam2gripper)
            print("t_cam2gripper:", self.t_cam2gripper)
            #Create 4x4 transfromation matrix
            self.T_cam2gripper = np.concatenate((self.R_cam2gripper, self.t_cam2gripper), axis=1)
            self.T_cam2gripper = np.concatenate((self.T_cam2gripper, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_cam2gripper_Method_{i}.npz", self.T_cam2gripper)
            #Save the inverse transfrom too
            self.T_gripper2cam = np.linalg.inv(self.T_cam2gripper)
            np.savez(f"cal_data/002/FinalTransforms/T_gripper2cam_Method_{i}.npz", self.T_gripper2cam)

        #solve hand-eye calibration using calibrateRobotWorldHandEye
        for i in range(0,2):
            self.R_base2world, self.t_base2world, self.R_gripper2cam, self.t_gripper2cam= cv2.calibrateRobotWorldHandEye( self.RTarget2Cam, self.TTarget2Cam, self.REE2Base, self.tEE2Base, method=i)
            #print and save each results as .npz file
            print("The results for method using calibrateRobotWorldHandEye", i+4, "are:")
            print("R_cam2gripper:", self.R_gripper2cam)
            print("t_cam2gripper:", self.t_gripper2cam)
            #Create 4x4 transfromation matrix T_gripper2cam
            self.T_gripper2cam = np.concatenate((self.R_gripper2cam, self.t_gripper2cam), axis=1)
            self.T_gripper2cam = np.concatenate((self.T_gripper2cam, np.array([[0, 0, 0, 1]])), axis=0)
            #Save results in folder FinalTransforms
            np.savez(f"FinalTransforms/T_gripper2cam_Method_{i+4}.npz", self.T_gripper2cam)
            #save inverse tooVecTose3
            self.T_cam2gripper = np.linalg.inv(self.T_gripper2cam)
            np.savez(f"cal_data/002/FinalTransforms/T_cam2gripper_Method_{i+4}.npz", self.T_cam2gripper)

    def find_chessboard_corners(self, images, pattern_size, ShowCorners=False,charuco_use=False):
        """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
        chessboard_corners = []
        IndexWithImg = []
        i = 0
        print("Finding corners...")
        if(charuco_use == False):
            for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, pattern_size)
                if ret:
                    chessboard_corners.append(corners)
                    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                    if ShowCorners:
                            #plot image using maplotlib. The title should "Detected corner in image: " + i
                        plt.imshow(image)
                        plt.title("Detected corner in image: " + str(i))
                        plt.show()
                #Save the image in a folder Named "DetectedCorners"
                #make folder
                    if not os.path.exists("DetectedCorners"):
                        os.makedirs("DetectedCorners")

                    cv2.imwrite("cal_data/001/DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                    IndexWithImg.append(i)
                    i = i + 1
                else:
                    print("No chessboard found in image: ", i)
                    i = i + 1
            ids_all = []
            return chessboard_corners, IndexWithImg, ids_all
        else:
            # ChAruco board variables
            # Corners discovered in all images processed
            corners_all = []
            # Aruco ids corresponding to corners discovered 
            ids_all = [] 
            for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_size = gray.shape[::-1]
                #print(image_size)
                corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=ARUCO_DICT)
                if ids is None:
                    continue
                response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=gray, board=CHARUCO_BOARD)	
                if response > 20:
                    # Add these corners and ids to our calibration arrays
                    corners_all.append(charuco_corners)
                    ids_all.append(charuco_ids)
                    # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                    img = aruco.drawDetectedCornersCharuco(image=image, charucoCorners=charuco_corners, charucoIds=charuco_ids)
                    if ShowCorners:
                        #plot image using maplotlib. The title should "Detected corner in image: " + i
                        plt.imshow(img)
                        plt.title("Detected corner in image: " + str(i))
                        plt.show()
                    if not os.path.exists("DetectedCorners"):
                        os.makedirs("DetectedCorners")

                    cv2.imwrite("cal_data/002/DetectedCorners/DetectedCorners" + str(i) + ".png", image)

                    IndexWithImg.append(i)
                    i = i + 1
                else:
                    print("No chessboard found in image: ", i)
                    i = i + 1
            return corners_all, IndexWithImg, ids_all
                



    def compute_camera_poses(self, chessboard_corners, pattern_size, square_size, intrinsic_matrix, Testing=False, charuco_use=False, ids_all=None ):
        """Takes the chessboard corners and computes the camera poses"""
        # Create the object points.Object points are points in the real world that we want to find the pose of.
        RTarget2Cam = []
        TTarget2Cam = []
        i = 1
        if charuco_use:
            for corners,ids in zip(chessboard_corners,ids_all):
                ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, CHARUCO_BOARD, intrinsic_matrix, None, np.empty(1),np.empty(1))
                i = 1 + i
                R, _ = cv2.Rodrigues(p_rvec)  # R is the rotation matrix from the target frame to the camera frame
                RTarget2Cam.append(R)
                TTarget2Cam.append(p_tvec)
            
            return RTarget2Cam, TTarget2Cam
        object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # Estimate the pose of the chessboard corners

        for corners in chessboard_corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)
            # rvec is the rotation vector, tvec is the translation vector
            if Testing == True:
                print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")

                # Convert the rotation vector to a rotation matrix
                print("rvec: ", rvec)
                print("rvec[0]: ", rvec[0])
                print("rvec[1]: ", rvec[1])
                print("rvec[2]: ", rvec[2])
                print("--------------------")
            i = 1 + i
            R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame
            RTarget2Cam.append(R)
            TTarget2Cam.append(tvec)

        return RTarget2Cam, TTarget2Cam

    def calculate_intrinsics(self, chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
        """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""
        # Find the corners of the chessboard in the image
        imgpoints = chessboard_corners
        # Find the corners of the chessboard in the real world
        objpoints = []
        for i in range(len(IndexWithImg)):
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
        # Find the intrinsic matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)

        print("The projection error from the calibration is: ",
              self.calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowProjectError))
        return mtx
    
    def calculate_intrinsics_charuco(self, chessboard_corners, ids_all, pattern_size, square_size, ImgSize, ShowProjectError=False):
        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(charucoCorners=chessboard_corners,charucoIds=ids_all,board=CHARUCO_BOARD,imageSize=(640, 480),cameraMatrix=None,distCoeffs=None)
        return cameraMatrix

    def calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
        """Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
        If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

        total_error = 0
        num_points = 0
        errors = []

        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            errors.append(error)
            total_error += error
            num_points += 1

        mean_error = total_error / num_points

        if ShowPlot:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            img_indices = range(1, len(errors) + 1)
            ax.bar(img_indices, errors)
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Reprojection Error')
            ax.set_title('Reprojection Error for Each Image')
            plt.show()
            print(errors)

            #Save the bar plot as a .png
            fig.savefig('ReprojectionError.png')

        return mean_error


def take_pho():

    clear_folder("cal_data/002/RGBImgs/")
    clear_folder("cal_data/002/T_base2ee/")

    real_robot = Yuil_robot()
    #real_robot.go_home(80)
    vid = cv2.VideoCapture(0) 
    start_time = time.time()
    run_num = 0.0
    #pos = [0.651, -0.123, 0.277,-3.14, 0.0, 1.56]
    pos = [0.601, -0.173, 0.277,-3.14, 0.0, 1.56]
    real_robot.xyz_move(pos,100)
    while(run_num < 50): 
        ret, frame = vid.read() 
        if (time.time() - start_time)%5 < 0.05 and run_num < (time.time() - start_time)/5:
            print((time.time() - start_time))
            pos_n = real_robot.robot_get_current_xyz_position()
            #print(pos_n[0],pos_n[1],pos_n[2],pos_n[3],pos_n[4],pos_n[5])
            pos_v = np.array([pos_n[0],pos_n[1],pos_n[2],pos_n[3],pos_n[4],pos_n[5]])
            pos_j = real_robot.robot_get_current_position()
            pos_j = np.array([pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5]])
            print(pos_j[0],pos_j[1],pos_j[2],pos_j[3],pos_j[4],pos_j[5])
            run_num = run_num +1
            if ret == True:
                cv2.imwrite("cal_data/002/RGBimgs/color_image" + str(int(run_num)).zfill(3) + ".png", frame)
                t2b_R,t2b_T = gripper2base(pos_v[3],pos_v[4],pos_v[5],pos_v[0]/1000,pos_v[1]/1000,pos_v[2]/1000)
                t2b_R = euler2rotm(np.array([pos_v[3],pos_v[4],pos_v[5]]))
                t2b_TR = RpToTrans(t2b_R,t2b_T)
                print(t2b_TR)
                np.savez_compressed("cal_data/002/T_base2ee/TBase2EE_" + str(int(run_num)).zfill(3) , t2b_TR)
                np.savez_compressed("cal_data/002/JointPositions/JointPositions" + str(int(run_num)).zfill(3) , pos_j)
                np.savez_compressed("cal_data/002/XYZPositions/XYZPositions" + str(int(run_num)).zfill(3) , pos_v)

            if int(run_num)%10 != 0:
                pos[0] = pos[0] + 0.01
                pos[1] = pos[1] + 0.01
                pos[3] = pos[3] + 0.01
                pos[4] = pos[4] + 0.01
                pos[5] = pos[5] + 0.01
            else:
                pos[0] = 0.601
                pos[1] = -0.173
                pos[2] = pos[2] + 0.05
                pos[3] = -3.14
                pos[4] = 0.0
                pos[5] = 1.56
                
            real_robot.xyz_move(pos,80)

        if ret==True:
            cv2.imshow('RGB', frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break
    pos = [0.601, -0.173, 0.277,-3.14, 0.0, 1.56]
    real_robot.xyz_move(pos,80)

    vid.release() 
    cv2.destroyAllWindows() 

def cal():
    image_folder = "cal_data/002/RGBimgs/"
    PoseFolder = "cal_data/002/T_base2ee/"
    calib = CameraCalibration(image_folder, PoseFolder,ShowProjectError=True,charuco_use=True)
def clear_folder(folder):
    files = glob.glob(folder+"*")
    print(files)
    for f in files:
        os.remove(f)
def main():
    #take_pho()
    cal()

if __name__ == "__main__":
    main()