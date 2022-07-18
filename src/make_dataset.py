# coding: utf-8

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob 
import cv2
import numpy as np
import torchvision.transforms as transforms


import os

class ImageFolder(Dataset):
    def __init__(self, img_dir, edge_dir, transform=None, num = 0):
        self.S_img_paths = self._get_img_paths(edge_dir, num)
        self.I_img_paths = self._get_img_paths(img_dir, num)

        #pathの順番がバラバラに読み込まれており、ペアが対応していないため並べる
        self.S_img_paths.sort()
        self.I_img_paths.sort()

        self.transform = transform

    def __getitem__(self, index):
        S_path = self.S_img_paths[index]
        I_path = self.I_img_paths[index]
        S_img = Image.open(S_path)
        I_img = Image.open(I_path)


        if self.transform is not None:
            S_img = self.transform(S_img)
            I_img = self.transform(I_img)

        return S_img, I_img


    def _get_img_paths(self, dir, num):
        img_paths = glob.glob(dir + '/**')
        if num != 0:
            img_paths = [img_paths[n] for n in range(num)]

        return img_paths
    
    def __len__(self):
        return len(min(self.S_img_paths, self.I_img_paths))
