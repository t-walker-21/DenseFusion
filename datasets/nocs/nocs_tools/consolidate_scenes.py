"""

Tool to grab all scenes from each scene folder and place into one folder
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np

# Seven different scenes
NUM_SCENES = 7

def read_intrinsics(path):
    """
    Function to read camera intrinsics from file and return in numpy array

    """

    fin = open(path)

    print (path)

    # Read info
    line_1 = fin.readlines()[2]
    fin.seek(0)
    line_2 = fin.readlines()[3]
    fin.close()

    line_1 = line_1.split(" ")
    line_2 = line_2.split(" ")

    intrinsics = np.zeros((3, 3))
    intrinsics[0][0] = line_1[0]
    intrinsics[0][2] = line_1[2]
    intrinsics[1][1] = line_2[1]
    intrinsics[1][2] = line_2[2]
    intrinsics[2][2] = 1

    return intrinsics

def read_gt_pose(path, index):
    """
    Read gt pose from label file and return in numpy array

    """

    # Open gt pose file
    fin = open(path)

    # Skip to desired index
    for _ in range(index * 4):
        fin.readline()

    pose = []

    # Extract pose
    for _ in range(4):
        pose.append(fin.readline().split(" ")[:-1])
    
    fin.close()

    return np.array(pose, dtype=np.float)

def gather_files(path, object_name):

    first = True

    # Keep count of images
    image_count = 0

    # Correct file path for glob
    if path[-1] != '/':
        path += '/'

    path += '*'

    # Move all images to consolidated folder (images are 640 x 480)
    for fold_num, folder in enumerate(glob.glob(path)):
        if 'all_scenes' in folder:
            continue

        for i in range(int(len(glob.glob(folder + "/*color.png")))):

            np.save(("{:s}all_scenes/{:04d}_pose.npy").format(path[:-1], image_count), read_gt_pose(folder + "/pose.txt", i))

            command = ("cp {:s}/{:04d}_color.png {:s}all_scenes/{:04d}_color.png".format(folder, i, path[:-1], image_count))
            print (command)
            os.system(command)

            command = ("cp {:s}/{:04d}_depth.png {:s}all_scenes/{:04d}_depth.png".format(folder, i, path[:-1], image_count))
            print (command)
            os.system(command)

            command = ("cp {:s}/{:04d}_meta.txt {:s}all_scenes/{:04d}_meta.txt".format(folder, i, path[:-1], image_count))
            print (command)
            os.system(command)

            command = ("cp {:s}/{:04d}_coord.png {:s}all_scenes/{:04d}_coord.png".format(folder, i, path[:-1], image_count))
            print (command)
            os.system(command)

            command = ("cp {:s}/{:04d}_mask.png {:s}all_scenes/{:04d}_mask.png".format(folder, i, path[:-1], image_count))
            print (command)
            os.system(command)
            image_count += 1



parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', type=str, required=False, help='object name of interest')
parser.add_argument('--NOCS_path', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/real_train', help='path for NOCS files')

opt = parser.parse_args()

gather_files(opt.NOCS_path, opt.obj_name)