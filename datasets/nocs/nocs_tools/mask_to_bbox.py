"""
Script to extract a bbox from a mask image

"""

import numpy as np
import cv2
import numpy.ma as ma
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder_path', type=str, required=True, help='path to consolidated dataset')
parser.add_argument('--object_id', type=int, required=False, help='object id for mask')
parser.add_argument('--object_keyword', type=str, required=True, help='object keyword for mask')
parser.add_argument('--job', type=str, required=True, help='task to do')
opt = parser.parse_args()

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
image_to_model = {}

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0

    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

def get_mask_from_raw_image(raw_mask, mask_id):
    return np.array(ma.getmaskarray(ma.masked_equal(raw_mask, mask_id))) * 100.0

def test_bbox(img, bbox):
    """
    Draw bbox on image to ensure correct bbox

    """

    img_bbox = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1, 1)

    cv2.imshow("test", img)
    cv2.waitKey(1)

def cam_in_image(file_path):
    """
    Based on meta file, determine if camera is in image

    """

    file_num = file_path.split("/")[-1][:4]

    try:
        fin = open(file_path)

    except:
        #print (file_path)
        #print ("Has incomplete entry")
        return 57, 47

    line = fin.readline()
    while line:
        if opt.object_keyword in line:
            if "camera" in opt.object_keyword:
                print ("debug")
                if "ana" in line:
                    image_to_model[file_num] = 1
                
                elif "wo_len" in line:
                    image_to_model[file_num] = 2

                elif "len" in line:
                    image_to_model[file_num] = 3

            fin.close()
            return True, int(line[0])
        
        line = fin.readline()
    fin.close()
    return False, 6

if opt.image_folder_path[-1] != '/':
    opt.image_folder_path += '/'

if opt.job == "bbox_proc":
    for i in range(4318):
        fname = opt.image_folder_path + "{:04d}_{:s}mask.png".format(i, opt.object_keyword)
        fname_color = opt.image_folder_path + "{:04d}_color.png".format(i)
        img = cv2.imread(fname)
        img_color = cv2.imread(fname_color)
        if img is None:
            continue

        bbox = mask_to_bbox(img[:, :, 0])

        np.save(opt.image_folder_path + "{:04d}_bbox.npy".format(i), bbox)

        #print (bbox)
        test_bbox(img_color, bbox)

elif opt.job == "sep":
    for i in range(4318):
        fname = opt.image_folder_path + "{:04d}_meta.txt".format(i)
        res, num = cam_in_image(fname)
        print (opt.image_folder_path + "{:04d}_mask.png".format(i))
        if res:
            # Object of interest is in this file
            try:
                img = cv2.imread(opt.image_folder_path + "{:04d}_mask.png".format(i))[:, :, 0]

            except:
                continue

            img_masked = get_mask_from_raw_image(img, num)

            cv2.imwrite(opt.image_folder_path + "{:04d}_{:s}mask.png".format(i, opt.object_keyword), img_masked)

            cv2.imshow("processed", img_masked)
            cv2.waitKey(1)

    print (image_to_model)
    np.save("image_to_model_dict.npy", image_to_model)