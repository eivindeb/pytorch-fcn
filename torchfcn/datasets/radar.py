#!/usr/bin/env python

import collections
import os.path as osp
import json

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from .radar_dataloader import data_loader
import datetime

class RadarClassSegBase(data.Dataset):  # why not generator-function?

    class_names = np.array([
        "ship",
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])  # TODO: what is this?

    def __init__(self, root, radar_type="Radar1", split='train', transform=False, dataset_name="radar_base"):
        self.root = root
        self.radar_type = radar_type
        self.split = split
        self._transform = transform
        self.dataset_name = dataset_name
        self.files = collections.defaultdict(list)
        datasets_dir = osp.join(self.root, "datasets")

        try:
            with open(osp.join(datasets_dir, 'config.json'), 'r') as f:
                cfg = json.load(f)
        except:
            print('Unable to read configuration file preprocess.json')
            cfg = None

        self.data_loader = data_loader(self.root, cfg)

        try:
            with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (self.dataset_name, self.radar_type, self.split)), "r") as file:
                for line in file:
                    line = line.strip()
                    self.files[split].append(line)

        except:
            print("No file found for %s dataset %s, generating file instead" % (self.split, self.dataset_name))
            self.generate_dataset_file(self.dataset_name)
            with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (self.dataset_name, self.radar_type, self.split), "r")) as file:
                for line in file:
                    line = line.strip()
                    self.files[split].append(line)

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
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img += self.mean_bgr
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    def generate_dataset_file(self, dataset_name):
        datasets_dir = osp.join(self.root, "datasets")
        with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (dataset_name, self.radar_type, self.split), "w")) as file:
            for hour in range(11, 13):
                for minute in range(0, 5):
                    for second in range(0, 60):
                        t = datetime.datetime(2017, 10, 12, hour, minute, second)
                        filename = self.data_loader.get_filename_sec(t, self.radar_type, "bmp")  # TODO: very slow
                        if len(filename) != 0:  # TODO: can there be more than one?
                            file.write("%s\n" % (filename[0]))

    def get_mean(self):
        mean_sum = 0
        for file in self.files[self.split]:
            img = self.data_loader.load_image(file)
            mean_sum += np.sum(img)/np.size(img)
        return mean_sum/len(self.files[self.split])


class RadarTest(RadarClassSegBase):

    def __init__(self, root, split='train', transform=False, dataset_name="radartest"):
        super(RadarTest, self).__init__(
            root, split=split, transform=transform, dataset_name=dataset_name)

#test = RadarTest("/media/stx/LaCie/export", split="train")
#print(test.get_mean())


