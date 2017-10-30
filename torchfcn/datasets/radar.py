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
from os import listdir, makedirs
import re
import random


class RadarDatasetFolder(data.Dataset):  # why not generator-function?

    class_names = np.array([
        "background",
        "ship",
    ])
    mean_bgr = np.array([50.3374548706, 50.3374548706, 50.3374548706])

    def __init__(self, root, split='train', transform=False, dataset_name="radar_base", radar_type="Radar1", cfg=None):
        self.root = root
        self.radar_type = radar_type
        self.split = split
        self._transform = transform
        self.dataset_name = dataset_name
        self.files = collections.defaultdict(list)
        datasets_dir = osp.join(self.root, "datasets")

        if cfg is not None:
            try:
                with open(cfg, 'r') as f:
                    cfg = json.load(f)
            except:
                print("Could not read config file {}".format(cfg))
                cfg = None

        self.data_loader = data_loader(self.root, cfg)

        try:
            if not osp.exists(datasets_dir):
                makedirs(datasets_dir)

            with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (self.dataset_name, self.radar_type, self.split)), "r") as file:
                for line in file:
                    line = line.strip()
                    self.files[split].append(line)

        except:
            print("No file found for %s dataset %s, generating file instead" % (self.split, self.dataset_name))
            self.generate_dataset_file()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img = self.data_loader.load_image(data_file)
        img_3ch = np.zeros((img.shape[0], 1000, 3))
        img_3ch[:, :, 0] = img[:, 0:1000]
        img_3ch[:, :, 1] = img[:, 0:1000]
        img_3ch[:, :, 2] = img[:, 0:1000]
        # load label
        lbl = self.data_loader.load_ais_layer(data_file.replace(".bmp", ".json"), img_3ch.shape[1], img_3ch.shape[0], 1, 1)
        if len(lbl) == 0:
            print("lbl zero")
            lbl = np.zeros(img_3ch.shape[0:2], dtype=np.int32)
        #  lbl[lbl == 255] = -1  TODO: why?
        if self._transform:
            return self.transform(img_3ch, lbl)
        else:
            return img_3ch, lbl

    def transform(self, img, lbl):
        # TODO: maybe we have to convert to RGB here?
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
        lbl = lbl.numpy()
        return img, lbl

    def generate_dataset_file(self):
        datasets_dir = osp.join(self.root, "datasets")
        files = self.collect_data_files_recursively(self.root)
        random.shuffle(files)
        with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (self.dataset_name, self.radar_type, "train")), "w+") as train:
            with open(osp.join(datasets_dir, "%s_%s_%s.txt" % (self.dataset_name, self.radar_type, "valid")), "w+") as valid:
                for i, filename in enumerate(files):
                    if i <= len(files)*0.8:
                        train.write("{}\n".format(filename))
                        self.files["train"].append(filename)
                    else:
                        valid.write("{}\n".format(filename))
                        self.files["valid"].append(filename)

    def collect_data_files_recursively(self, parent, files=None):
        if files is None:
            files = []

        for child in listdir(parent):
            if re.match("^[0-9-]*$", child) or child == self.radar_type:
                files = self.collect_data_files_recursively(osp.join(parent, child), files)
            elif child.endswith(".bmp"):
                files.append(osp.join(parent, child))

        return files

    def get_mean(self):
        mean_sum = 0
        for file in self.files[self.split]:
            img = self.data_loader.load_image(file)
            mean_sum += np.sum(img)/np.size(img)
        return mean_sum/len(self.files[self.split])


class RadarTest(RadarDatasetFolder):

    def __init__(self, root, split='train', transform=False, dataset_name="radartest", cfg=None):
        super(RadarTest, self).__init__(
            root, split=split, transform=transform, dataset_name=dataset_name, cfg=cfg)


#config = osp.expanduser("~/Projects/sensorfusion/logging/preprocess.json")
#test = RadarTest("/media/stx/LaCie/export/2017-10-12", split="valid", cfg=config)
#print("get")
#print(test.get_mean())


