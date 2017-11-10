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


"""
TODO:
- load chart layer and use to set land as unlabeled?
- choose one data file from each 10 minute block
- remove targets which are not visible to radar (hidden by land in front). Perform bitwise OR on columns to create mask of everything hidden by land
- finish caching labels
- address class imbalance (perhaps in loss function?)
- modify visualization function (show percentage estimation of class?)
"""
here = osp.dirname(osp.abspath(__file__))


class RadarDatasetFolder(data.Dataset):

    class_names = np.array([
        "background",
        "ship",
    ])

    class_weights = np.array([
        1,
        11000,
    ])

    mean_bgr = np.array([50.3374548706, 50.3374548706, 50.3374548706])
    INDEX_FILE_NAME = "{}_{}_{}.txt"

    def __init__(self, root, dataset_name, split='train', transform=False, radar_type=("Radar1", "Radar0"),
                 data_ranges=(np.s_[:, :]), cache_labels=False, filter_land=False,
                 land_is_target=False, remove_hidden_targets=True, min_data_interval=0,
                 remove_files_without_targets=True):
        self.root = root
        self.radar_type = radar_type
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

        try:
            if not osp.exists(self.datasets_dir):
                makedirs(self.datasets_dir)

            with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, self.split)), "r+") as file:
                lines = file.readlines()
                file_edited = False
                for line_num, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="Reading dataset index file"):
                    line_edited = False
                    line = line.strip()
                    filename, ranges = line.split(";")
                    if self.remove_files_without_targets:
                        ranges_with_targets, ranges_without_targets = ranges.split("/")

                        for i, data_range in enumerate(self.data_ranges):
                            if str(data_range) in ranges_with_targets or not self.remove_files_without_targets:
                                self.files[split].append([filename, i])
                            elif str(data_range) in ranges_without_targets:
                                continue
                            else:  # new data range, check for targets in range and write result to index file
                                line_edited = True
                                edit_pos = line.rfind("/")

                                if ais is None:
                                    ais, land = self.get_labels(filename)

                                lbl = ais[data_range].astype(dtype=np.uint8)

                                if land is not None:
                                    lbl[land[data_range] == 1] = 2 if self.land_is_target else 0

                                if np.max(lbl) > 0:
                                    self.files[split].append([filename, i])
                                    line = line[:edit_pos] + str(data_range) + line[edit_pos:]
                                else:
                                    line = line[:edit_pos + 1] + str(data_range) + line[edit_pos + 1:]
                    else:
                        for i in range(len(self.data_ranges)):
                            self.files[self.split].append([filename, i])

                    if line_edited:
                        file_edited = True
                        lines[line_num] = line

                    ais, label = None, None

                if file_edited:
                    file.writelines(lines)

        except:
            print("No index file found for dataset, generating files instead")
            self.generate_dataset_file()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index][0]
        data_range = self.data_ranges[self.files[self.split][index][1]]

        # load image
        img = self.data_loader.load_image(data_file)

        # Construct 3 channel version of image from selected data range
        img_3ch = np.repeat(img[data_range[0], data_range[1], np.newaxis], 3, axis=2)

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

        sorted_files = sorted(files, key=lambda x: datetime.datetime.strptime(x.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S_%f"))
        last_time = datetime.datetime(year=2000, month=1, day=1)

        for file in tqdm.tqdm(sorted_files, total=len(sorted_files), desc="Filtering data files", leave=False):
            file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S_%f")

            if file_time - last_time < datetime.timedelta(minutes=self.min_data_interval):
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
                    last_time = file_time
                else:
                    filter_stats["No targets"] += 1
            else:
                filtered_files.append([file, self.data_ranges])

        print("{} data files left after filtering (time: {}, no targets: {})".format(len(filtered_files), filter_stats["Time"], filter_stats["No targets"]))

        print("Writing to index file")

        random.shuffle(filtered_files)

        with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, "train")), "w+") as train:
            with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(self.dataset_name, self.radar_type, "valid")), "w+") as valid:
                for i, file in enumerate(filtered_files):
                    checked_ranges = ""
                    for j in file[1]:
                        checked_ranges += str(self.data_ranges[j])
                    checked_ranges += "/"
                    for j in file[2]:
                        checked_ranges += str(self.data_ranges[j])

                    if i <= len(filtered_files)*0.8:
                        train.write("{};{}\n".format(file[0], checked_ranges))
                        if self.split == "train":
                            for j in file[1]:
                                self.files["train"].append([file[0], j])
                    else:
                        valid.write("{};{}\n".format(file[0], checked_ranges))
                        if self.split == "valid":
                            for j in file[1]:
                                self.files["valid"].append([file[0], j])

    def get_mean(self):
        mean_sum = 0
        for file in self.files[self.split]:
            img = self.data_loader.load_image(file[0])
            mean_sum += np.mean(img, dtype=np.float64)
        return mean_sum/len(self.files[self.split])

    def get_class_shares(self):
        class_shares = {c: 0 for c in self.class_names}

        filter_land_prev = self.filter_land
        self.filter_land = False

        for i, file in tqdm.tqdm(enumerate(self.files[self.split]), total=len(self.files[self.split]),
                                 desc="Calculating class shares for {} data".format(self.split), leave=False):
            radar_index = int(file.split("/")[-2][-1])

            ais, land = self.data_loader.load_ais_layer(file.replace(".bmp", ".json"), 1, radar_index)

            for j, data_range in enumerate(self.data_ranges):
                lbl = ais[data_range].astype(dtype=np.uint8)

                if land is not None:
                    lbl[land[data_range] == 1] = 2

                for c_index, c in enumerate(self.class_names):
                    class_shares[c] = (class_shares[c] + lbl[lbl == c_index].size/lbl.size)/(2 if i + j != 0 else 1)

        self.filter_land = filter_land_prev

        return class_shares

    def collect_and_cache_labels(self):
        files = self.files[self.split]

        with open(osp.join(self.datasets_dir, self.INDEX_FILE_NAME.format(
                self.dataset_name, self.radar_type, "train" if self.split == "valid" else "valid")), "r") as file:
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

            if self.filter_land:
                try:
                    land = np.load(data_file.replace(".bmp", "_label_land_hidden.npy"))
                except IOError as e:
                    land = self.data_loader.load_chart_layer(meta_file, 0, 1, radar_index)
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


class RadarShipTargetFilterLandAndHidden(RadarDatasetFolder):

    def __init__(self, root, split='train', transform=False, dataset_name="radartest",
                 data_ranges=(np.s_[:int(4096/2), 0:2000], np.s_[int(4096/2):, 0:2000]), cache_labels=True,
                 min_data_interval=0):

        super(RadarShipTargetFilterLandAndHidden, self).__init__(root, split=split, transform=transform,
            dataset_name=dataset_name, data_ranges=data_ranges, cache_labels=cache_labels, filter_land=True,
            land_is_target=False, remove_hidden_targets=True, remove_files_without_targets=True,
            min_data_interval=min_data_interval)


#valid = RadarShipTargetFilterLandAndHidden("/media/stx/LaCie1/export/", split="valid", dataset_name="no_time_filter")
#b = valid[0]
#print("hei")


