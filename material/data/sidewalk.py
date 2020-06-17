#!/usr/bin/env python
import os
import collections
import os.path as osp
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from ..utils import image_transform as it


class SidewalkClassSeg(data.Dataset):
    class_names = np.array([
        'Unlabeled',
        'Sidewalk',
    ])

    def __init__(self, dataset_dir, split=['train'], transform=0):
        self.dataset_dir = dataset_dir
        self.split = split
        self._transform = transform
        self.files = []

        if 'train' in split:
            start = 0
            end = 15000
        else:
            start = 15000
            end = 20000

        tar_img_dir = osp.join(dataset_dir, 'test')
        tar_lbl_dir = osp.join(dataset_dir, 'test_seg')  # gtFine

        for image in os.listdir(tar_img_dir)[start:end]:

            image_name = image.split(".")[0]

            img_file = osp.join(tar_img_dir, image_name + ".jpg")
            lbl_file = osp.join(tar_lbl_dir, image_name + ".png")

            if image.split(".")[1] not in ["jpg"]:
                continue

            self.files.append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        lbl_file = data_file['lbl']
        img = it.process_img_file(img_file)
        lbl = it.process_lbl_file(lbl_file)

        img_tensor, lbl_tensor = it.to_tensor(img, lbl)

        return it.transform(img_tensor, lbl_tensor, self._transform, 'sidewalk')
