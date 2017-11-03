#!/usr/bin/env python

import collections
import os.path as osp
import json

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from radar_dataloader import data_loader
import datetime
from os import listdir, makedirs
import re
import random


"""
TODO:
- load chart layer and use to set land as unlabeled?
- remove targets which are not visible to radar (hidden by land in front)
- finish caching labels
- address class imbalance (perhaps in loss function?)
- modify visualization function (show percentage estimation of class?)
- should probably cache entire chart layer
"""


class RadarDatasetFolder(data.Dataset):  # why not generator-function?

    class_names = np.array([
        "background",
    ])

    class_weights = np.array([
        1,
        11000,
    ])

    mean_bgr = np.array([50.3374548706, 50.3374548706, 50.3374548706])
    INDEX_FILE_NAME = "{}_{}_{}.txt"

    def __init__(self, root, split='train', transform=False, dataset_name="radar_base", radar_type="Radar1",
                 data_range=np.s_[:, :], cache_labels=False, filter_land=False, targets=["ship"], cfg=None):
        self.root = root
        self.radar_type = radar_type
        self.split = split
        self._transform = transform
        self.dataset_name = dataset_name
        self.data_range = data_range
        self.cache_labels = cache_labels
        self.filter_land = filter_land
        self.targets = targets
        self.files = collections.defaultdict(list)
        datasets_dir = osp.join(self.root, "datasets")

        if "land" in targets and filter_land:
            print("Land is both a target and filtered out, exiting")
            exit(0)

        for target in targets:
            np.append(self.class_names, target)

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

            with open(osp.join(datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, self.split)), "r") as file:
                for line in file:
                    line = line.strip()
                    self.files[split].append(line)

        except:
            print("No index file found for dataset, generating files instead")
            self.generate_dataset_file()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        def get_cached_label_name(data_range, data_size, targets, land_filtered):
            label_name = ""
            if data_range[0].start is None and data_range[0].stop is None and data_range[0].step is None:
                label_name += "_labels_all"
            else:
                label_name += "_labels_{}-{}-{}".format(data_range[0].start if data_range[0].start is not None else 0,
                                                        data_range[0].step if data_range[0].step is not None else 1,
                                                        data_range[0].stop if data_range[0].stop is not None else data_size[0])
            if data_range[1].start is None and data_range[1].stop is None and data_range[1].step is None:
                label_name += "_all"
            else:
                label_name += "_{}-{}-{}".format(data_range[1].start if data_range[1].start is not None else 0,
                                                data_range[1].step if data_range[1].step is not None else 1,
                                                data_range[1].stop if data_range[1].stop is not None else data_size[1])

            for target in targets:
                label_name += "_{}".format(target)

            if land_filtered:
                label_name += "_land-filtered"

            return label_name

        data_file = self.files[self.split][index]
        # load image
        img = self.data_loader.load_image(data_file)

        # Construct 3 channel version of image from selected data range
        img_3ch = np.repeat(img[self.data_range[0], self.data_range[1], np.newaxis], 3, axis=2)

        # load label
        cached_label_missing = False
        if self.cache_labels:
            try:
                lbl = np.load(data_file.replace(".bmp", "{}.npy".format(get_cached_label_name(self.data_range, img.shape, self.targets, self.filter_land ))))
            except IOError as e:
                cached_label_missing = True

        if not self.cache_labels or cached_label_missing:
            meta_file = data_file.replace(".bmp", ".json")
            if "ship" in self.targets:
                lbl = self.data_loader.load_ais_layer(meta_file, img.shape[1], img.shape[0], 1, 1).astype(dtype=np.int32)
                if len(lbl) == 0:
                    lbl = np.zeros(img_3ch.shape[0:2], dtype=np.int32)
                else:
                    lbl = lbl[self.data_range]
            else:
                lbl = np.zeros(img_3ch.shape[0:2], dtype=np.int32)

            if "land" in self.targets or self.filter_land:
                land = self.data_loader.load_chart_layer(meta_file, img.shape[1], img.shape[0], 0, 1, 1)[self.data_range]

                lbl[land == 1] = 2 if "land" in self.targets else -1

            if cached_label_missing:
                np.save(data_file.replace(".bmp", get_cached_label_name(self.data_range, img.shape, self.targets, self.filter_land)), lbl)

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

    def generate_dataset_file(self, remove_files_without_targets=True):
        def no_targets(progress, filename):
            print("Checking targets for image {}".format(progress))
            img = self.data_loader.load_image(filename)
            targets = self.data_loader.load_ais_layer(filename.replace(".bmp", ".json"), img.shape[1], img.shape[0], 1, 1)
            return len(targets) == 0 or np.max(targets[self.data_range]) == 0

        def collect_data_files_recursively(parent, filenames=None):
            if filenames is None:
                filenames = []

            for child in listdir(parent):
                if re.match("^[0-9-]*$", child) or child == self.radar_type:
                    filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                elif child.endswith(".bmp"):
                    filenames.append(osp.join(parent, child))

            return filenames

        datasets_dir = osp.join(self.root, "datasets")
        files = collect_data_files_recursively(self.root)
        if remove_files_without_targets:
            files = [file for i, file in enumerate(files) if not no_targets("{}/{}".format(i, len(files)), file)]
        random.shuffle(files)
        with open(osp.join(datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, "train")), "w+") as train:
            with open(osp.join(datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, "valid")), "w+") as valid:
                for i, filename in enumerate(files):
                    if i <= len(files)*0.8:
                        train.write("{}\n".format(filename))
                        self.files["train"].append(filename)
                    else:
                        valid.write("{}\n".format(filename))
                        self.files["valid"].append(filename)

    def get_mean(self):
        mean_sum = 0
        for file in self.files[self.split]:
            img = self.data_loader.load_image(file)
            mean_sum += np.sum(img)/np.size(img)
        return mean_sum/len(self.files[self.split])

    def get_class_shares(self):
        class_shares = {c: 0 for c in self.class_names}
        for i, file in enumerate(self.files[self.split]):
            print("Calculating class shares for file {} of {}".format(i, len(self.files[self.split])))
            img = self.data_loader.load_image(file)
            lbl = self.data_loader.load_ais_layer(file.replace(".bmp", ".json"), img.shape[1], img.shape[0], 1, 1)
            for c_index, c in enumerate(self.class_names):
                class_shares[c] = (class_shares[c] + lbl[lbl == c_index].size/lbl.size)/(2 if i != 0 else 1)

        return class_shares


class RadarTest(RadarDatasetFolder):

    def __init__(self, root, split='train', transform=False, dataset_name="radartest",
                 data_range=np.s_[200:3800, 0:1000], cache_labels=True, targets=["ship"], filter_land=True, cfg=None):
        super(RadarTest, self).__init__(
            root, split=split, transform=transform, dataset_name=dataset_name,
            data_range=data_range, cache_labels=cache_labels, targets=targets, filter_land=filter_land, cfg=cfg)


config = osp.expanduser("~/Projects/sensorfusion/logging/preprocess.json")
valid = RadarTest("/media/stx/LaCie/export/", split="valid", cfg=config)
train = RadarTest("/media/stx/LaCie/export/", split="train", cfg=config)
test = [valid, train]
#print(test.get_class_shares())'
for dataset in test:
    for i, fi in enumerate(dataset.files[dataset.split]):
        junk = dataset[i]
        print("Cached labels for {} ({}/{})".format(fi, i, len(dataset)))
#print("get")
#print(test.get_mean())


