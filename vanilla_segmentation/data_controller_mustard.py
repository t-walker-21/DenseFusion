import torch
import numpy as np
from PIL import Image
import numpy.ma as ma
import torch.utils.data as data
import copy
from torchvision import transforms
import scipy.io as scio
import torchvision.datasets as dset
import random
import scipy.misc
import scipy.io as scio
import os
from PIL import ImageEnhance
from PIL import ImageFilter
import cv2

class SegDataset(data.Dataset):
    def __init__(self, root_dir, txtlist, use_noise, length):
        self.path = []
        self.use_noise = use_noise
        self.root = root_dir
        input_file = open(txtlist)

        image_width = 1280
        image_height = 720

        print ("dataroot: ", self.root)

        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.path.append(copy.deepcopy(input_line))
            if input_line[:5] == 'data/':
                self.path.append(copy.deepcopy(input_line))
        input_file.close()

        self.length = len(self.path)
        self.data_len = len(self.path)

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.back_front = np.array([[1 for i in range(image_width)] for j in range(image_height)])

    def __getitem__(self, idx):
        index = random.randint(0, self.data_len)

        label = np.array(Image.open('{0}mask/{1}-label.png'.format(self.root, self.path[index])))

        if not self.use_noise:
            rgb = np.array(Image.open('{0}rgb/{1}-color.png'.format(self.root, self.path[index])).convert("RGB"))
        
        else:
            rgb = np.array(self.trancolor(Image.open('{0}rgb/{1}-color.png'.format(self.root, self.path[index])).convert("RGB")))
    
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.fliplr(rgb)
                rgb = np.flipud(rgb)
                label = np.fliplr(label)
                label = np.flipud(label)
                
        target = copy.deepcopy(label)


        #cv2.imshow("color", rgb)
        #cv2.imshow("label", target)

        #cv2.waitKey(0)

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = self.norm(torch.from_numpy(rgb.astype(np.float32)))
        target = torch.from_numpy(target.astype(np.int64))

        return rgb, target


    def __len__(self):
        return self.length

