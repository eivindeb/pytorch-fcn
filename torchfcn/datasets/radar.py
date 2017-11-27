#!/usr/bin/env python

import collections
import os.path as osp
import json

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from .dataloader import data_loader
import datetime
from os import listdir, makedirs
import re
import random
import tqdm
from math import exp


"""
TODO:
- modify visualization function (show percentage estimation of class?)
"""


class DataRange:
    def __init__(self, data_range):
        self.data_range = self.fill_None(data_range)


    def in_range(self, data_range):
        pass

    def fill_None(self, data_range):
        for axis in data_range:
            if axis.stop is None:
                axis.stop = 0
            if axis.start is None:
                axis.start = -1
            if axis.step is None:
                axis.step = 1

        return data_range

#r = DataRange(np.s_[0:1000, :2000])
#r.fill_None(r.data_range)



here = osp.dirname(osp.abspath(__file__))


class RadarDatasetFolder(data.Dataset):

    class_names = np.array([
        "background",
        "ship",
    ])

    class_weights = np.array([  # based on frequency of targets in data
        1,
        14700,
    ])

    mean_bgr = np.array([58.61890545754149, 58.61890545754149, 58.61890545754149])
    INDEX_FILE_NAME = "{}_{}_{}.txt"

    def __init__(self, root, dataset_name, split='train', transform=False, radar_type=("Radar1", "Radar0"),
                 data_ranges=(np.s_[:, :]), cache_labels=False, filter_land=False,
                 land_is_target=False, remove_hidden_targets=True, min_data_interval=0,
                 remove_files_without_targets=True):
        self.root = root
        self.radar_type = radar_type  # TODO: fix printing when not tuple
        self.split = split
        self._transform = transform
        self.dataset_name = dataset_name
        self.data_ranges = data_ranges
        self.cache_labels = cache_labels
        self.filter_land = filter_land
        self.land_is_target = land_is_target
        self.files = collections.defaultdict(list)
        self.remove_hidden_targets = remove_hidden_targets
        self.remove_files_without_targets = remove_files_without_targets
        self.min_data_interval = min_data_interval
        self.datasets_dir = osp.join(self.root, "datasets")

        if land_is_target:
            if filter_land:
                print("Exiting: Land is both declared as a target and filtered out, please adjust dataset settings.")
                exit(0)
            else:
                np.append(self.class_names, "land")

        self.data_loader = data_loader(self.root, sensor_config=osp.join(here, "dataloader.json"))

        # TODO: fix path connections between location of dataset index,and data files and labels

        try:
            if not osp.exists(self.datasets_dir):
                makedirs(self.datasets_dir)

            with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, "-".join(self.radar_type), self.split)), "r+") as file:
                lines = file.readlines()
                file_edited = False
                ais = None
                for line_num, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="Reading dataset index file"):
                    line_edited = False
                    line = line.strip()
                    filename, ranges = line.split(";")
                    ranges_with_targets, ranges_without_targets = ranges.split("/")

                    for i, data_range in enumerate(self.data_ranges):
                        if not self.remove_files_without_targets or str(data_range) in ranges_with_targets:
                            self.files[split].append([osp.join(self.root, filename), i])
                        elif str(data_range) in ranges_without_targets:
                            continue
                        else:  # new data range, check for targets in range and write result to index file
                            line_edited = True
                            edit_pos = line.rfind("/")

                            if ais is None:
                                ais, land = self.get_labels(osp.join(self.root, filename))

                            lbl = ais[data_range].astype(dtype=np.uint8)

                            if land is not None:
                                lbl[land[data_range] == 1] = 2 if self.land_is_target else 0

                            if np.max(lbl) > 0:
                                self.files[split].append([osp.join(self.root, filename), i])
                                line = line[:edit_pos] + str(data_range) + line[edit_pos:]
                            else:
                                line = line[:edit_pos + 1] + str(data_range) + line[edit_pos + 1:]

                    if line_edited:
                        file_edited = True
                        lines[line_num] = line+"\n"

                    ais, label = None, None

                if file_edited:
                    file.seek(0)
                    file.truncate()
                    file.writelines(lines)

        except IOError as e:
            print(e)
            print("No index file found for dataset, generating index for dataset instead")
            self.generate_dataset_file()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index][0]
        data_range = self.data_ranges[self.files[self.split][index][1]]

        # load image
        img = self.data_loader.load_image(data_file)[data_range]


        # Construct 3 channel version of image with exponential and linear mask to model noise intensity with range
        img_3ch = np.empty((img.shape[0], img.shape[1], 3))
        img_3ch[:, :, 0] = img

        max_val = 255
        min_val = 0
        interval_length = max_val - min_val
        decay_constant = -10 / img.shape[1]

        for col in range(img.shape[1]):
            img_3ch[:, col, 1] = int(interval_length * exp(decay_constant * col) + min_val)
            img_3ch[:, col, 2] = int(-((max_val - min_val) / (img.shape[1] - 1)) * col + max_val)

        # load label
        ais, land = self.get_labels(data_file)

        lbl = ais[data_range].astype(dtype=np.int32)

        if land is not None:
            lbl[land[data_range] == 1] = 2 if self.land_is_target else -1

        if self._transform:
            return self.transform(img_3ch, lbl)
        else:
            return img_3ch, lbl

    def transform(self, img, lbl):
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
        def collect_data_files_recursively(parent, filenames=None):
            if filenames is None:
                filenames = []

            for child in listdir(parent):
                if re.match("^[0-9-]*$", child) or child in self.radar_type:
                    filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                elif child.endswith(".bmp"):
                    filenames.append(osp.join(parent, child))

            return filenames

        files = collect_data_files_recursively(self.root)
        print("Found {} data files".format(len(files)))

        filtered_files = []
        filter_stats = {"Time": 0, "No targets": 0}

        sorted_files = sorted(files, key=lambda x: datetime.datetime.strptime(x.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S"))
        last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_type}

        for file in tqdm.tqdm(sorted_files, total=len(sorted_files), desc="Filtering data files", leave=False):
            file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
            file_radar_type = file.split("/")[-2]

            if file_time - last_time[file_radar_type] < datetime.timedelta(minutes=self.min_data_interval):
                filter_stats["Time"] += 1
                continue

            if self.remove_files_without_targets:
                ais, land = self.get_labels(file)

                lbl = ais.astype(dtype=np.uint8)

                if land is not None:
                    if len(land) == 0:
                        continue
                    lbl[land == 1] = 2 if self.land_is_target else 0

                ranges_with_targets = []
                ranges_without_targets = []

                for j in range(0, len(self.data_ranges)):
                    if np.max(lbl[self.data_ranges[j]]) > 0:
                        ranges_with_targets.append(j)
                    else:
                        ranges_without_targets.append(j)

                if len(ranges_with_targets) > 0:
                    filtered_files.append([file, ranges_with_targets, ranges_without_targets])
                    last_time[file_radar_type] = file_time
                else:
                    filter_stats["No targets"] += 1
            else:
                filtered_files.append([file, self.data_ranges])

        print("{} data files left after filtering (time: {}, no targets: {})".format(len(filtered_files), filter_stats["Time"], filter_stats["No targets"]))

        print("Writing to index file")

        random.shuffle(filtered_files)

        with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, "-".join(self.radar_type), "train")), "w+") as train:
            with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, "-".join(self.radar_type), "valid")), "w+") as valid:
                for i, file in enumerate(filtered_files):
                    checked_ranges = ""
                    for j in file[1]:
                        checked_ranges += str(self.data_ranges[j])
                    checked_ranges += "/"
                    for j in file[2]:
                        checked_ranges += str(self.data_ranges[j])

                    filename = osp.relpath(file[0], start=self.root)

                    if i <= len(filtered_files)*0.8:
                        train.write("{};{}\n".format(filename, checked_ranges))
                        if self.split == "train":
                            for j in file[1]:
                                self.files["train"].append([file[0], j])
                    else:
                        valid.write("{};{}\n".format(filename, checked_ranges))
                        if self.split == "valid":
                            for j in file[1]:
                                self.files["valid"].append([file[0], j])

    def get_mean(self):
        mean_sum = 0
        for file in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]),
                              desc="Calculating mean for dataset"):
            img = self.data_loader.load_image(file[0])
            data_range = self.data_ranges[file[1]]
            mean_sum += np.mean(img[data_range], dtype=np.float64)
        return mean_sum/len(self.files[self.split])

    def get_class_shares(self):
        class_shares = {c: 0 for c in self.class_names}

        for i, file in tqdm.tqdm(enumerate(self.files[self.split]), total=len(self.files[self.split]),
                                 desc="Calculating class shares for {} data".format(self.split), leave=False):
            ais, land = self.get_labels(file[0])
            data_range = self.data_ranges[file[1]]

            lbl = ais[data_range].astype(dtype=np.uint8)

            if land is not None:
                lbl[land[data_range] == 1] = 2

            for c_index, c in enumerate(self.class_names):
                class_shares[c] += lbl[lbl == c_index].size/lbl.size

        class_shares.update({c: class_shares[c]/len(self.files[self.split]) for c in class_shares.keys()})

        return class_shares

    def collect_and_cache_labels(self):
        files = self.files[self.split]

        with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(
                self.dataset_name, "-".join(self.radar_type), "train" if self.split == "valid" else "valid")), "r") as file:
            for line in file:
                line = line.strip()
                files.append(line.split(";"))

        for i, f in enumerate(files):
            print("Caching labels for file {} of {}".format(i, len(files)))
            ais, land = self.get_labels(f[0])

    def get_labels(self, data_file):
        radar_index = int(data_file.split("/")[-2][-1])  # Extract radar type from filename e.g. 0 from ../Radar0/filename.bmp
        meta_file = data_file.replace(".bmp", ".json")

        if self.cache_labels:
            try:
                ais = np.load(data_file.replace(".bmp", "_label_ship.npy"))
            except IOError as e:
                ais = self.data_loader.load_ais_layer(meta_file, 1, radar_index)

                if len(ais) == 0:
                    img = self.data_loader.load_image(data_file)
                    ais = np.zeros(img.shape, dtype=np.int32)
                else:
                    np.save(data_file.replace(".bmp", "_label_ship"), ais)

            if self.land_is_target:
                try:
                    land = np.load(data_file.replace(".bmp", "_label_land.npy"))
                except IOError as e:
                    land = self.data_loader.load_chart_layer(meta_file, 0, 1, radar_index)
                    if len(land) != 0:
                        np.save(data_file.replace(".bmp", "_label_land"), land)

            if self.filter_land:  # TODO: fix self.remove_hidden_targets (that is, ships) when land is target
                try:
                    land = np.load(data_file.replace(".bmp", "_label_land_hidden.npy"))
                except IOError as e:
                    if not self.land_is_target:
                        try:
                            land = np.load(data_file.replace(".bmp", "_label_land.npy"))
                        except IOError as e:
                            land = self.data_loader.load_chart_layer(meta_file, 0, 1, radar_index)
                            if len(land) != 0:
                                np.save(data_file.replace(".bmp", "_label_land"), land)

                    if len(land) != 0:
                        hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
                        hidden_by_land_mask[:, 0] = land[:, 0]
                        for col in range(1, land.shape[1]):
                            np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])

                        np.save(data_file.replace(".bmp", "_label_land_hidden"), hidden_by_land_mask)
                        land = hidden_by_land_mask

        else:
            ais = self.data_loader.load_ais_layer(meta_file, 1, radar_index)

            if len(ais) == 0:
                img = self.data_loader.load_image(data_file)
                ais = np.zeros(img.shape, dtype=np.int32)

            if self.land_is_target or self.filter_land:
                land = self.data_loader.load_chart_layer(meta_file, 0, 1, radar_index)

                if self.filter_land and len(land) != 0:
                    hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
                    hidden_by_land_mask[:, 0] = land[:, 0]
                    for col in range(1, land.shape[1]):
                        np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])

                    land = hidden_by_land_mask

        if self.filter_land or self.land_is_target:
            return ais, land
        else:
            return ais, None

    def generate_list_of_required_files(self):
        index_file = self.INDEX_FILE_NAME.format(self.dataset_name, "-".join(self.radar_type), self.split)

        required_files = index_file.replace(".txt", "_required_files.txt")

        sorted_files = sorted(self.files[self.split], key=lambda x: datetime.datetime.strptime(x[0].split("/")[-1].replace(".bmp", ""),
                                                                              "%Y-%m-%d-%H_%M_%S"))

        with open(osp.join(self.root, required_files), "w+") as file:
            last_filename = ""
            for data_file in sorted_files:
                data_file = osp.relpath(data_file[0], start=self.root)

                if data_file == last_filename:
                    continue

                lines = [data_file + "\n", data_file.replace(".bmp", ".json\n"), data_file.replace(".bmp", ".txt\n")]
                if self.cache_labels:
                    lines.append(data_file.replace(".bmp", "_label_ship.npy\n"))
                    if self.land_is_target or (self.filter_land and self.remove_hidden_targets):
                        lines.append(data_file.replace(".bmp", "_label_land.npy\n"))
                    if self.remove_hidden_targets:
                        lines.append(data_file.replace(".bmp", "_label_land_hidden.npy\n"))

                last_filename = data_file
                file.writelines(lines)


class RadarShipTargetFilterLandAndHidden(RadarDatasetFolder):

    def __init__(self, root, split='train', transform=True, dataset_name="radartest",
                 data_ranges=(np.s_[:int(4096/3), 0:2000], np.s_[int(4096/3):int(2*4096/3), 0:2000], np.s_[int(2*4096/3):, 0:2000]), cache_labels=True,
                 min_data_interval=0):

        super(RadarShipTargetFilterLandAndHidden, self).__init__(root, split=split, transform=transform,
            dataset_name=dataset_name, data_ranges=data_ranges, cache_labels=cache_labels, filter_land=True,
            land_is_target=False, remove_hidden_targets=True, remove_files_without_targets=True,
            min_data_interval=min_data_interval)


#valid = RadarShipTargetFilterLandAndHidden("/media/stx/LaCie1/export/", split="train", dataset_name="no_time_filter")
#print(valid.get_mean())
#valid.generate_list_of_required_files()
#print("hei")
#for i in tqdm.tqdm(range(len(valid.files[valid.split])), total=len(valid.files[valid.split])):
#    img, data = valid[i]

#print("Training set mean: {}".format(train.get_mean()))
#print("Training set class shares: {}".format(train.get_class_shares()))
#test = np.load("/media/stx/LaCie1/export/2017-10-25/2017-10-25-17/Radar0/2017-10-25-17_00_02_513_label_land.npy")
#print(test)


