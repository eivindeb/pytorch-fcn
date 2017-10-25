#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from radar_dataloader import data_loader
import datetime


class RadarClassSegBase(data.Dataset):  # why not generator-function?

    class_names = np.array([
        "ship",
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])  # TODO: what is this?

    def __init__(self, root, radar_type="Radar1", split='train', transform=False):
        self.root = root
        self.radar_type = radar_type
        self.split = split
        self._transform = transform
        self.data_manager = data_loader(self.root)

        dataset_dir = self.root  #osp.join(self.root, '')
        self.files = collections.defaultdict(list)

        # TODO: maybe just load from file instead
        for hour in range(0, 2):
            for minute in range(0, 5):
                for second in range(0, 60):
                    t = datetime.datetime(2017, 10, 12, hour, minute, second)
                    filename = self.data_manager.get_filename_sec(t, self.radar_type, "bmp")  # TODO: very slow
                    if len(filename) != 0:  # TODO: can there be more than one?
                        if minute < 55:
                            self.files["train"].append(t)
                        else:
                            self.files["valid"].append(t)


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img = self.data_manager.load_image(data_file, 1, 1)
        # load label
        lbl = self.data_manager.load_ais_layer(data_file, img.shape[0], img.shape[1], 1, 1)
        #  lbl[lbl == 255] = -1  TODO: why?
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class RadarTest(RadarClassSegBase):

    def __init__(self, root, split='train', transform=False):
        super(RadarTest, self).__init__(
            root, split=split, transform=transform)

test = RadarClassSegBase("/media/stx/LaCie/")
test.__getitem__(132)
print(test)


