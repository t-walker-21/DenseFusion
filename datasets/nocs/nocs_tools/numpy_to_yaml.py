"""
Script to convert pose labels from numpy files to yaml format

"""

import numpy as np
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder_path', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/real_train/all_scenes', help='path to consolidated dataset')
parser.add_argument('--gt_file', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/real_train/all_scenes/gt.yaml', help='name for output file')
opt = parser.parse_args()

def read_pose(file_path):
    """
    Read pose from numpy file

    """

    # Read pose
    pose = np.load(file_path)

    rotation = pose[:3, :3]
    translation = pose[:3, -1]
    return rotation, translation

def read_bbox(file_path):
    """
    Read bbox from numpy file

    """
    
    # Read and return bbox
    return np.load(file_path)

if opt.image_folder_path[-1] != "/":
    opt.image_folder_path += "/"

fout = open(opt.gt_file, "w")

data_points = {}
f_count = 0

train_list = open("train.txt", "w")

for i in range(4317):
    fname_pose = opt.image_folder_path + "{:04d}_pose.npy".format(i)
    fname_bbox = opt.image_folder_path + "{:04d}_bbox.npy".format(i)

    rot, trans = read_pose(fname_pose)
    try:
        bbox = read_bbox(fname_bbox)

    except:
        continue
    
    train_list.write("{:04d}\n".format(i))
    print (f_count)
    f_count += 1

    data_point = {}

    # Camera rotation
    data_point['cam_R_m2c'] = rot.flatten().tolist()

    # Camera translation
    data_point['cam_t_m2c'] = trans.flatten().tolist()

    # Object id
    data_point['obj_id'] = 100

    data_point['obj_bb'] = bbox.flatten().tolist()

    data_points[i] = data_point


yaml.dump(data_points, fout)

fout.close()
train_list.close()
exit()