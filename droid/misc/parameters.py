import os
from cv2 import aruco

# Robot Params #
nuc_ip = None
robot_ip = "192.168.1.13"
laptop_ip = "127.0.0.1"
sudo_password = "3363"
robot_type = "fr3" # 'panda' or 'fr3'
robot_serial_number = ""

# Camera ID's #
hand_camera_id = ""
varied_camera_1_id = ""
varied_camera_2_id = ""

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = "C121Cqc9bAm4rzZkCvZzuBdtGFQYU2"

# Code Version [DONT CHANGE] #
droid_version = "1.3"