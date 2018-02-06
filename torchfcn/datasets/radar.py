#!/usr/bin/env python

import collections
import os.path as osp
import json

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import datetime
from os import listdir, makedirs, remove
import re
import random
import tqdm
from configparser import ConfigParser
import shutil
import subprocess


"""
TODO:
- modify visualization function (show percentage estimation of class?)
- make data loader continually check for new files from nas
- move datasetfolder to root argument, write label and data folder to config file maybe along with parameters
- calculate mean bgr in generate dataset and write to config
- finish DataRange class or find smarter way to write what parts of image has targets
- add weight to log dir name
"""


class MaxDiskUsageError(Exception):
    pass


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
    class_weights = np.array([  # based on frequency of targets in data
        1,
        5000,
    ])

    # mean_bgr = np.array([55.9, 55.9, 56])
    #mean_bgr = np.array([55.1856378125, 55.1856378125, 53.8775])
    mean_bgr = np.array([55.1856378125])
    INDEX_FILE_NAME = "{}_{}_{}.txt"
    LABELS = {"background": 0, "ais": 1, "land": 2, "hidden": 3, "unlabeled": -1}

    def __init__(self, root, dataset_name, cfg, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.files = collections.defaultdict(list)
        self.dataset_folder = osp.join(self.root, dataset_name)
        if not osp.exists(self.dataset_folder):
            makedirs(self.dataset_folder)

        config = ConfigParser()
        try:
            with open(cfg, 'r') as cfg:
                config.read_file(cfg)
        except IOError:
            print("Configuration file not found at {}".format(cfg))
            exit(0)
        self.cache_labels = config["Parameters"].getboolean("CacheLabels", False)
        self.data_folder = config["Paths"].get("DataFolder")
        self.label_folder = config["Paths"].get("LabelFolder")

        if self.data_folder is None:
            print("Configuration file missing required field: DataFolder")
            exit(0)

        if self.cache_labels and self.label_folder is None:
            print("Cache labels is set to true, but no label folder is provided.")
            exit(0)

        self.radar_types = [radar for radar in config["Parameters"].get("RadarTypes", "Radar0,Radar1,Radar2").split(",")]
        self.filter_land = config["Parameters"].getboolean("FilterLand", False)
        self.remove_hidden_targets = config["Parameters"].getboolean("RemoveHiddenTargets", True)
        self.class_names = np.array([c for c in config["Parameters"].get("Classes", "background,ship").split(",")])
        self.class_weights = np.array([int(weight) for weight in config["Parameters"].get("ClassWeights", "1,5000").split(",")])
        self.remove_files_without_targets = config["Parameters"].getboolean("RemoveFilesWithoutTargets", True)
        self.min_data_interval = config["Parameters"].getint("MinDataIntervalSeconds", 0)
        self.skip_processed_files = config["Parameters"].getboolean("SkipProcessedFiles", True)
        self.coordinate_system = config["Parameters"].get("CoordinateSystem", "Polar")
        self.max_disk_usage = config["Parameters"].get("MaxDiskUsage", None)

        try:
            self.max_disk_usage = int(self.max_disk_usage)
        except:
            print("Config parameter MaxDiskUsage must be an integer of bytes allowed.")
            exit(0)

        if self.cache_labels and self.max_disk_usage is None:
            print("Warning: No maximum disk usage specified, using all available space.")
            exit(0)

        disk_usage = shutil.disk_usage(self.label_folder)
        if self.max_disk_usage > disk_usage.free:
            print("Warning: maximum allowed disk usage is larger than the available disk space.")

        self.current_disk_usage = subprocess.check_output(["du", "-sx", "/data/polarlys/labels"]).split()[0].decode("utf-8")

        self.max_disk_capacity_reached = {"status": disk_usage.free < 1e6, "timestamp": datetime.datetime.now()}

        self.data_ranges = (np.s_[:int(4096/3), 0:2000], np.s_[int(4096/3):int(2*4096/3), 0:2000], np.s_[int(2*4096/3):, 0:2000])

        if "land" in self.class_names and self.filter_land:
            print("Exiting: Land is both declared as a target and filtered out, please adjust configuration.")
            exit(0)

        self.data_loader = DataLoader(self.data_folder, sensor_config=osp.join(here, "dataloader.json"))

        # TODO: fix path connections between location of dataset index,and data files and labels
        try:
            self.load_files_from_index(osp.join(self.dataset_folder, self.split) + ".txt")
        except IOError as e:
            print(e)
            print("No index file found for dataset, generating index for dataset instead")
            self.generate_dataset_file()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_path = self.files[self.split][index]["data"][0]
        data_range = self.data_ranges[self.files[self.split][index]["data"][1]]

        # load image
        basename = osp.splitext(data_path)[0]
        t = self.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

        img = self.data_loader.load_image(t, sensor, sensor_index)
        # load label
        lbl = self.get_label(data_path, self.files[self.split][index]["label"])

        if self.coordinate_system == "Cartesian":
            cart = self.data_loader.transform_image_from_sensor(t, sensor, sensor_index, dim=2828, scale=3.39,
                                                               image=np.dstack((img, lbl)).astype(np.int16), use_gpu=True)
            img = cart[:, :, 0].astype(np.uint8)
            lbl = cart[:, :, 1].astype(np.int8)
        img = img[data_range]
        lbl = lbl[data_range]
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = np.squeeze(img, axis=0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    def load_files_from_index(self, index):
        with open(index, "r+") as file:
            lines = file.readlines()
            file_edited = False
            lbl = None
            for line_num, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="Reading dataset index file"):
                line_edited = False
                line = line.strip()
                filename, ranges = line.split(";")
                ranges_with_targets, ranges_without_targets = ranges.split("/")

                # img = self.data_loader.load_image(osp.join(self.data_folder, filename))

                # if img is None:  # temporary
                #    line_edited = True
                #    line = "removed"
                if False:
                    pass
                else:
                    for i, data_range in enumerate(self.data_ranges):
                        if not self.remove_files_without_targets or str(data_range) in ranges_with_targets:
                            self.files[self.split].append({"data": [osp.join(self.data_folder, filename), i],
                                                      "label": osp.join(self.label_folder,
                                                                        filename.replace(".bmp", "_label.npy"))})
                        elif str(data_range) in ranges_without_targets:
                            continue
                        else:  # new data range, check for targets in range and write result to index file
                            line_edited = True
                            edit_pos = line.rfind("/")

                            if lbl is None:
                                lbl = self.get_label(osp.join(self.data_folder, filename),
                                                     osp.join(self.label_folder,
                                                              filename.replace(".bmp", "_label.npy")))

                            if np.any(lbl[data_range] == self.LABELS["ais"]):
                                self.files[self.split].append({"data": [osp.join(self.data_folder, filename), i],
                                                          "label": osp.join(self.label_folder,
                                                                            filename.replace(".bmp", "_label.npy"))})
                                line = line[:edit_pos] + str(data_range) + line[edit_pos:]
                            else:
                                line = line[:edit_pos + 1] + str(data_range) + line[edit_pos + 1:]

                if line_edited:
                    file_edited = True
                    lines[line_num] = line + "\n"

                lbl = None

            if file_edited:
                file.seek(0)
                file.truncate()
                lines = [line for line in lines if line != "removed"]
                file.writelines(lines)

        if self.min_data_interval > 0:
            self.files[self.split] = sorted(self.files[self.split], key=lambda x: datetime.datetime.strptime(
                x["data"][0].split("/")[-1].replace(".bmp", ""),
                "%Y-%m-%d-%H_%M_%S"))
            last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_types}

            new_files = []

            for file_info in self.files[self.split]:
                file = file_info["data"][0]
                file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                file_radar_type = file.split("/")[-2]

                if file_time - last_time[file_radar_type] > datetime.timedelta(seconds=self.min_data_interval):
                    last_time[file_radar_type] = file_time
                    new_files.append(file_info)

            self.files[self.split] = new_files

    def generate_dataset_file(self):
        def collect_data_files_recursively(parent, filenames=None):
            if filenames is None:
                filenames = []

            for child in listdir(parent):
                if re.match("^[0-9-]*$", child) or child in self.radar_types:
                    filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                elif child.endswith(".bmp"):
                    filenames.append(osp.join(parent, child))

            return filenames

        files = collect_data_files_recursively(self.data_folder)
        print("Found {} data files".format(len(files)))

        filtered_files = []
        filter_stats = {"Time": 0, "No targets": 0}

        sorted_files = sorted(files, key=lambda x: datetime.datetime.strptime(x.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S"))
        sorted_files = sorted_files[302000:]


        with open(osp.join(self.root, "processed_files.txt"), "r+") as processed_files_index:
            lines = processed_files_index.readlines()
            files_without_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] == "false\n"]
            files_with_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] == "true\n"]

            #last_processed = lines[-1].rstrip("\n").split(";")[0].replace(".bmp", "")
            #last_processed_time = datetime.datetime.strptime(last_processed, "%Y-%m-%d-%H_%M_%S")
            last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_types}
            for file in tqdm.tqdm(sorted_files, total=len(sorted_files), desc="Filtering data files", leave=False):
                try:
                    if self.skip_processed_files and (self.remove_files_without_targets and file in files_without_targets or file in files_with_targets):
                        continue

                    file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                    file_radar_type = file.split("/")[-2]

                    if file_time - last_time[file_radar_type] < datetime.timedelta(seconds=self.min_data_interval):
                        filter_stats["Time"] += 1
                        continue

                    img = self.data_loader.load_image(file)

                    if img is None or type(img) == "list":  # temporary
                        continue

                    if self.remove_files_without_targets:
                        if file in files_without_targets:
                            continue

                        try:
                            lbl = self.get_label(file, file.replace(self.data_folder, self.label_folder).replace(".bmp", "_label.npy"), throw_exception=True)
                        except MaxDiskUsageError:
                            print("Maximum allowed disk usage reached, terminating file search")
                            break
                        except OSError:
                            print("Maximum disk capacity reached, terminating file search")
                            break

                        ranges_with_targets = []
                        ranges_without_targets = []

                        for j in range(0, len(self.data_ranges)):
                            if np.any(lbl[self.data_ranges[j]] == self.LABELS["ais"]):
                                ranges_with_targets.append(j)
                            else:
                                ranges_without_targets.append(j)

                        if len(ranges_with_targets) > 0:
                            filtered_files.append([file, ranges_with_targets, ranges_without_targets])
                            last_time[file_radar_type] = file_time
                            processed_files_index.write("{};true\n".format(file))
                        else:
                            processed_files_index.write("{};false\n".format(file))
                            files_without_targets.append(file)

                            label_path = file.replace(self.data_folder, self.label_folder).replace(".bmp", "_label.npy")
                            remove(label_path)

                            filter_stats["No targets"] += 1
                    else:
                        filtered_files.append([file, self.data_ranges])
                except Exception as e:  # temporary
                    print("An error occurred in processing of image {}, skipping".format(file))
                    print(e)
                    processed_files_index.write("{};false\n".format(file))
                    files_without_targets.append(file)
                    continue

        print("{} data files left after filtering (time: {}, no targets: {})".format(len(filtered_files), filter_stats["Time"], filter_stats["No targets"]))

        print("Writing to index file")

        random.shuffle(filtered_files)

        with open(osp.join(self.dataset_folder, "train.txt"), "w+") as train:
            with open(osp.join(self.dataset_folder, "valid.txt"), "w+") as valid:
                for i, file in enumerate(filtered_files):
                    checked_ranges = ""
                    for j in file[1]:
                        checked_ranges += str(self.data_ranges[j])
                    checked_ranges += "/"
                    for j in file[2]:
                        checked_ranges += str(self.data_ranges[j])

                    file_rel_path = osp.relpath(file[0], start=self.data_folder)
                    label_path = osp.join(self.label_folder, file_rel_path).replace(".bmp", "_label.npy")

                    if i <= len(filtered_files)*0.8:
                        train.write("{};{}\n".format(file_rel_path, checked_ranges))
                        if self.split == "train":
                            for j in file[1]:
                                self.files["train"].append({"data": [file[0], j],
                                                           "label": label_path})
                    else:
                        valid.write("{};{}\n".format(file_rel_path, checked_ranges))
                        if self.split == "valid":
                            for j in file[1]:
                                self.files["valid"].append({"data": [file[0], j],
                                                           "label": label_path})

    def get_mean(self):
        mean_sum = 0
        missing_files = []
        for file in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]),
                              desc="Calculating mean for dataset"):
            file = file["data"]
            img = self.data_loader.load_image(file[0])
            if img is not None:
                data_range = self.data_ranges[file[1]]
                mean_sum += np.mean(img[data_range], dtype=np.float64)
            else:
                missing_files.append(file[0])
        return mean_sum/(len(self.files[self.split]) - len(missing_files))

    def get_mean_of_columns(self):
        mean_sum = np.zeros((1, 2000))
        missing_files = []

        for file in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]),
                              desc="Calculating mean for columns of dataset"):
            file = file["data"]
            img = self.data_loader.load_image(file[0])
            if img is not None:
                data_range = self.data_ranges[file[1]]
                mean_sum += np.mean(img[data_range], axis=0, dtype=np.float64)
            else:
                missing_files.append(file[0])

        return mean_sum/len(self.files[self.split])

    def get_class_shares(self):
        class_shares = {c: 0 for c in self.class_names}

        for i, file in tqdm.tqdm(enumerate(self.files[self.split]), total=len(self.files[self.split]),
                                 desc="Calculating class shares for {} data".format(self.split), leave=False):
            lbl_path = file["label"]
            file = file["data"]
            lbl = self.get_label(file[0], lbl_path)[:, 0:2000]
            data_range = self.data_ranges[file[1]]

            for c_index, c in enumerate(self.class_names):
                class_shares[c] += lbl[lbl == c_index].size/lbl.size

        class_shares.update({c: class_shares[c]/len(self.files[self.split]) for c in class_shares.keys()})

        return class_shares

    def save_numpy_file(self, file_path, file, throw_exception=False):
        if self.max_disk_capacity_reached and datetime.datetime.now() - self.max_disk_capacity_reached["timestamp"] < datetime.timedelta(hours=1):
            if throw_exception:
                raise OSError
        else:
            try:
                if self.current_disk_usage + file.nbytes > self.max_disk_usage:
                    if throw_exception:
                        raise MaxDiskUsageError
                else:
                    self.current_disk_usage += file.nbytes
                    np.save(file_path, file)
            except OSError:  # numpy cannot allocate enough free space
                if throw_exception:
                    raise OSError
                self.max_disk_capacity_reached["status"] = True
                self.max_disk_capacity_reached["timestamp"] = datetime.datetime.now()

    def collect_and_cache_labels(self):
        files = self.files[self.split]

        with open(osp.join(self.dataset_folder, "train" if self.split == "valid" else "valid"), "r") as file:
            for line in file:
                line = line.strip()
                files.append(line.split(";"))

        for i, f in enumerate(files):
            print("Caching labels for file {} of {}".format(i, len(files)))
            ais, land = self.get_label(f[0])

    def get_label(self, data_path, label_path, throw_exception=False):
        cached_label_missing = False
        if self.cache_labels:
            try:
                label = np.load(label_path).astype(np.int32)
            except IOError as e:
                cached_label_missing = True

        if not self.cache_labels or cached_label_missing:
            basename = osp.splitext(data_path)[0]
            t = self.data_loader.get_time_from_basename(basename)
            sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)
            ais = self.data_loader.load_ais_layer_sensor(t, sensor, sensor_index)

            if len(ais) == 0:
                img = self.data_loader.load_image(data_path)
                if len(img) == 0 or img is None:
                    label = np.zeros((4096, 3400), dtype=np.int32)
                else:
                    label = np.zeros(img.shape, dtype=np.int32)
            else:
                label = ais.astype(np.int8)

            if "land" in self.class_names or self.filter_land:
                try:
                    land = np.load(label_path.replace(".npy", "_land.npy"))
                except:
                    land = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index)
                label[land == 1] = self.LABELS["land"]

                if self.filter_land and len(land) != 0:
                    hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
                    hidden_by_land_mask[:, 0] = land[:, 0]
                    for col in range(1, land.shape[1]):
                        np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])

                    if self.remove_hidden_targets:
                        label[(hidden_by_land_mask == 1) & (land == 0)] = self.LABELS["hidden"]
                    else:
                        label[(hidden_by_land_mask == 1) & (land == 0) & (ais == 0)] = self.LABELS["hidden"]

            # unlabel data blocked by mast for Radar0
            if sensor_index == 0:
                label[2000:2080, :] = self.LABELS["unlabeled"]

            if cached_label_missing:
                if not osp.exists(osp.dirname(label_path)):
                    makedirs(osp.dirname(label_path))
                self.save_numpy_file(label_path, label.astype(np.int8), throw_exception=throw_exception)

        if self.filter_land:
            label[(label == self.LABELS["land"]) | (label == self.LABELS["hidden"])] = self.LABELS["unlabeled"]

        else:
            if "land" not in self.class_names:
                label[(label == self.LABELS["land"]) | (label == self.LABELS["hidden"])] = self.LABELS["background"]
            elif self.remove_hidden_targets:
                label[(label == self.LABELS["ais"]) & (label == self.LABELS["hidden"])] = self.LABELS["background"]

        return label

    def update_cached_labels(self, component):
        processed_labels = []
        for entry in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]), desc="Updating cached labels", leave=False):
            if entry["label"] in processed_labels:
                continue

            label_path = entry["label"]
            data_path = entry["data"][0]
            try:
                label = np.load(label_path).astype(np.int32)
            except IOError as e:
                print("Label does not exists at {}".format(label_path))
                continue

            if component in self.LABELS:
                label[label == self.LABELS[component]] = self.LABELS["background"]
            elif component == "chart":
                label[(label == self.LABELS["land"]) | (label == self.LABELS["hidden"])] = self.LABELS["background"]

            basename = osp.splitext(data_path)[0]
            t = self.data_loader.get_time_from_basename(basename)
            sensor, sensor_index, subsensor_index = self.data_loader.get_sensor_from_basename(basename)

            if component == "ais":
                ais = self.data_loader.load_ais_layer_sensor(t, sensor, sensor_index, subsensor_index)
                label[ais == 1] = self.LABELS["ais"]
            elif component == "chart":
                land = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index, subsensor_index)
                hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
                hidden_by_land_mask[:, 0] = land[:, 0]
                for col in range(1, land.shape[1]):
                    np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])

                label[(hidden_by_land_mask == 1) & (land == 0)] = self.LABELS["hidden"]
                label[land == 1] = self.LABELS["land"]

            # unlabel data blocked by mast for Radar0
            if sensor_index == 0:
                label[2000:2080, :] = self.LABELS["unlabeled"]

            self.save_numpy_file(label_path, label.astype(np.int8))
            processed_labels.append(label_path)


    def generate_list_of_required_files(self):
        index_file = osp.join(self.dataset_folder, self.split) + ".txt"

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
                    if "land" in self.class_names or (self.filter_land and self.remove_hidden_targets):
                        lines.append(data_file.replace(".bmp", "_label_land.npy\n"))
                    if self.remove_hidden_targets:
                        lines.append(data_file.replace(".bmp", "_label_land_hidden.npy\n"))

                last_filename = data_file
                file.writelines(lines)


if __name__ == "__main__":
    from dataloader import DataLoader

    #np.s_[:int(4096/3), 0:2000], np.s_[int(4096/3):int(2*4096/3), 0:2000], np.s_[int(2*4096/3):, 0:2000]
    valid = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets", cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt", split="train", dataset_name="2018")
    valid.update_cached_labels("ais")

    exit(0)
    print("mean: ")
    print(valid.get_mean())
    mean_cols = valid.get_mean_of_columns()
    np.savetxt("column_sum_new.txt", mean_cols)

    print("class shares: ")
    class_shares = valid.get_class_shares()
    print(class_shares)
    print("hei")
    #valid.generate_list_of_required_files()
    #print("hei")
    #for i in tqdm.tqdm(range(len(valid.files[valid.split])), total=len(valid.files[valid.split])):
    #    img, data = valid[i]

    #print("Training set mean: {}".format(train.get_mean()))
    #print("Training set class shares: {}".format(train.get_class_shares()))
    #test = np.load("/media/stx/LaCie1/export/2017-10-25/2017-10-25-17/Radar0/2017-10-25-17_00_02_513_label_land.npy")
    #print(test)
else:
    from .dataloader import DataLoader


