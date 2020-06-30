"""
Script to visualize and confirm labels in consolidated dataset

"""
import numpy as np
import cv2
import pyrender
import trimesh
import argparse

def visualize_prediction(pred_r, pred_t, mesh):
	"""
	Function to visualize prediction of object pose
	"""
	
	cam_fx = 577.5
	cam_fy = 577.5
	cam_cx = 319.5
	cam_cy = 293.5
	
	image_shape = (640, 480)
    
    #cam_cx = 312.9869
	#cam_cy = 241.3109
	#cam_fx = 1066.778
	#cam_fy = 1067.487

	cam = pyrender.IntrinsicsCamera(fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy)
    
	# Rotate prediction from normal coordinate system to openGL coordinate system

	openGL_T_normal = np.array([[1, 0, 0],
								[0, -1, 0],
								[0, 0, -1]])

	openGL_T_normal = np.linalg.inv(openGL_T_normal)

	pred_r = openGL_T_normal.dot(pred_r)

    # Build transformation matrix
	mat_pred = np.eye(4)
	mat_pred[:3, :3] = pred_r
	mat_pred[:3, 3] = pred_t


	# Invert y and z translations
	mat_pred[2, -1] *= -1
	mat_pred[1, -1] *= -1

	# Create light object
	light = pyrender.SpotLight(color=np.ones(3), intensity=10,
	                            innerConeAngle=np.pi/16.0,
	                            outerConeAngle=np.pi/6.0)

	# Create a scene
	scene = pyrender.Scene()

	# Create camera node object
	nc = pyrender.Node(camera=cam, matrix=np.eye(4))

	# Create object node object
	no_pred = pyrender.Node(mesh=mesh, matrix=mat_pred)

	# Create light node object
	nl = pyrender.Node(light=light, matrix=np.eye(4))

	# Add camera to scene
	scene.add_node(nc)

	# Add object to scene
	scene.add_node(no_pred, parent_node=nc)

	# Add light to scene
	scene.add_node(nl, parent_node=nc)

	# Create object renderer
	render = pyrender.OffscreenRenderer(image_shape[0], image_shape[1])

	# Render images
	color_pred, depth_pred = render.render(scene)

	# Convert color
	color_pred = cv2.cvtColor(color_pred, cv2.COLOR_BGR2RGB)

	cv2.imshow("pred", color_pred)
	cv2.waitKey(0)



parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/real_train/all_scenes', help='path to consolidated dataset')
parser.add_argument('--image_num', type=int, default=0, help='image num')
parser.add_argument('--object_meshes_path', type=str, default='/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/obj_models/real_train/', help='path to mesh')
parser.add_argument('--model_name', required=True, type=str, help='model name')

opt = parser.parse_args()

if opt.folder_path[-1] != "/":
    opt.folder_path += "/"

pose_file = "{:s}{:04d}_pose.npy".format(opt.folder_path, opt.image_num)

pose = np.load(pose_file)

if 'ana' in opt.model_name:
    mod_name = "/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/obj_models/real_train/camera_dslr_len_norm.obj"
	#mod_name = "/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/NOCS/obj_models/train/03797390/1eaf8db2dd2b710c7d5b1b70ae595e60/model.obj"
	#mod_name = "/media/tevon/b0926983-e0a6-464b-954c-71b0676b885c/tevonwalker/Documents/models/025_mug/textured.obj"

model = trimesh.load(mod_name)
mesh = pyrender.Mesh.from_trimesh(model, smooth=False)

scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)


pose = np.eye(4)
#pose[2][3] = -0.4

pose[2][3] = 0.4
pose[0][3] *= 0
pose[1][3] *= 0
print (pose[:3, 3])

visualize_prediction(pose[:3, :3], pose[:3, 3], mesh)