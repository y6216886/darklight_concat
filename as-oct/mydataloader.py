import torch.utils.data as data
import torchvision.datasets as dsets
from PIL import Image
import os
import os.path
import random
import numbers
import torchvision.transforms as transforms
import pickle
from os.path import exists, join
import sys
from os import makedirs
import numpy as np
import re
import pandas as pd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    isImage = False
    for extension in IMG_EXTENSIONS:
        if filename.endswith(extension):
            isImage = True
    return isImage


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def pil_loader(path):
    return Image.open(path).convert('RGB')




def make_dataset(dir_dark,dir_light, label_df):
    images = []
    index= 0
    itemlist = []
    for line1, line2 in zip(open(dir_dark), open(dir_light)):
        img_path_dark_org = line1.rstrip('\n')
        img_path_dark_org = img_path_dark_org.rstrip ('\r')
        img_path_dark = img_path_dark_org.split('/')[-1]
        # img_id = re.sub('.jpg', '', img_path.split('/')[-1])
        flag = img_path_dark.split("_")[1]
        img_id = img_path_dark.split("_")[0].split("-")[0]+"-"+img_path_dark.split("_")[0].split("-")[1]+"-"+str(int(img_path_dark.split("_")[-1].strip(".jpg")))

        ##
        img_path_light_org = line2.rstrip ('\n')
        img_path_light_org = img_path_light_org.rstrip ('\r')
        img_path_light = img_path_light_org.split ('/')[-1]

        if flag =="R":
            img_left_label = label_df.loc[img_id, 'od_left']
            img_right_label = label_df.loc[img_id, 'od_right']
        elif flag =="L":
            img_left_label = label_df.loc[img_id, 'os_left']
            img_right_label = label_df.loc[img_id, 'od_right']
        else:
            print("filename load error not Left or Right filename:", img_id)
        item = (img_path_dark_org, img_path_light_org, img_left_label, img_right_label)
        itemlist.append(item)
        if int(index) ==127:
            images.append(itemlist)
            itemlist=[]
            index = 0
            continue
        index += 1
    return images



def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('eyeId')
    return label_df

def padimg(img):##PIL Image
    img = np.array (img)
    h, w, _ = img.shape
    dim_diff = np.abs (h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    img = np.pad (img, pad, 'constant', constant_values=128)
    img = Image.fromarray (img.astype ('uint8')).convert ('RGB')
    return img

class Myloader(data.Dataset):
    def __init__(self, root_dark, root_light, label_dir, transform=None):
        self.root_dark = root_dark
        self.root_light = root_light
        self.label_dir = label_dir
        self.loader = pil_loader
        self.transform = transform

        self.label_df = get_label(label_dir)
        self.imgs = make_dataset(root_dark, root_light, self.label_df)


    def __getitem__(self, index):
        print("》》》》》self.imgs", self.imgs[index])
        for path_dark, path_light, left, right in self.imgs[index]:

            img_dark = self.loader(path_dark)
            img_dark= self.transform (img_dark)
            img_light = self.loader(path_light)
            img_light= self.transform (img_light)


            img = (img_dark, img_light)
            labels = (left, right)
        return img, labels

    def __len__(self):
        return len(self.imgs)