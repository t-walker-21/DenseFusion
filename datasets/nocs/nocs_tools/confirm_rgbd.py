"""
Script to visualize and confirm rgb d images

"""
import numpy as np
import cv2
import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/real_train/all_scenes', help='path to consolidated dataset')
parser.add_argument('--image_num', type=int, default=0, help='image num')

opt = parser.parse_args()

if opt.folder_path[-1] != "/":
    opt.folder_path += "/"

rgb_file = "{:s}{:04d}_color.png".format(opt.folder_path, opt.image_num)
depth_file = "{:s}{:04d}_depth.png".format(opt.folder_path, opt.image_num)

color_raw = o3d.io.read_image(rgb_file)
depth_raw = o3d.io.read_image(depth_file)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

# Init intrinsics
intrin = o3d.open3d.camera.PinholeCameraIntrinsic()

intrin.set_intrinsics(640, 380, 577, 577, 319, 239)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrin)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])