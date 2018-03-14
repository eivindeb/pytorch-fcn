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
import logging
import cv2
from matplotlib import pyplot as plt
from torchfcn import cc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fcn.utils import label2rgb


"""
TODO:
- modify visualization function (show percentage estimation of class?)
- make data loader continually check for new files from nas
- calculate mean bgr in generate dataset and write to config
- add weight to log dir name
"""


class WeatherDataMissing(Exception):
    pass


class MetadataNotFound(Exception):
    pass


class MissingStatsError(Exception):
    pass


class MaxDiskUsageError(Exception):
    pass


class LabelSourceMissing(Exception):
    pass


class DataFileNotFound(Exception):
    pass


class AmbiguousDataPath(Exception):
    pass


class RadarDatasetFolder(data.Dataset):
    LABEL_SOURCE = {"ais": 1, "chart": 2}
    LABELS = {"background": 0, "vessel": 1, "land": 2, "unknown": 3, "islet": 4, "unlabeled": -1}

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
        self.cache_folders = config["Paths"].get("CacheFolder(s)").split(",")

        if self.cache_labels and self.cache_folders is None:
            print("Cache labels is set to true, but no label folder is provided.")
            exit(0)

        for cache_folder in self.cache_folders:
            if not osp.exists(cache_folder):
                makedirs(cache_folder)

        dataloader_config = config["Paths"].get("DataloaderCFG")

        if dataloader_config is None:
            print("Configuration file missing required field: DataloaderCFG")
            exit(0)

        if self.data_folder is None:
            print("Configuration file missing required field: DataFolder")
            exit(0)

        self.radar_types = [radar for radar in config["Parameters"].get("RadarTypes", "Radar0,Radar1,Radar2").split(",")]
        self.unlabel_chart_data = config["Parameters"].getboolean("UnlabelChartData", False)
        self.remove_hidden_targets = config["Parameters"].getboolean("RemoveHiddenTargets", True)
        self.class_names = np.array([c for c in config["Parameters"].get("Classes", "background,ship").split(",")])
        self.class_weights = np.array([int(weight) for weight in config["Parameters"].get("ClassWeights", "1,5000").split(",")])
        self.remove_files_without_targets = config["Parameters"].getboolean("RemoveFilesWithoutTargets", True)
        self.min_data_interval = config["Parameters"].getint("MinDataIntervalSeconds", 0)
        self.skip_processed_files = config["Parameters"].getboolean("SkipProcessedFiles", True)
        self.coordinate_system = config["Parameters"].get("CoordinateSystem", "Polar")
        self.max_disk_usage = config["Parameters"].get("MaximumDiskUsage", None)
        self.set_splits = [float(s) for s in config["Parameters"].get("SetSplits", "0.95,0.025,0.025").split(",")]
        self.image_width = config["Parameters"].getint("ImageWidth", 2000)
        self.image_height = config["Parameters"].getint("ImageHeight", 4096)
        self.land_threshold = config["Parameters"].getint("LandThreshold", 70)
        self.include_weather_data = config["Parameters"].getboolean("IncludeWeatherData", False)
        self.chart_area_threshold = config["Parameters"].getint("ChartAreaThreshold", 10000)
        self.min_vessel_land_dist = config["Parameters"].getfloat("MinVesselLandDistance", 10)

        height_divisons = config["Parameters"].getint("HeightDivisions", 2)
        width_divisons = config["Parameters"].getint("WidthDivisions", 0)
        overlap = config["Parameters"].getint("Overlap", 20)

        self.set_data_ranges(height_divisons, width_divisons, overlap=overlap)

        if sum(self.set_splits) != 1:
            print("Desired set split does not sum to 1, instead {}".format(sum(self.set_splits)))
            exit(0)

        if self.cache_labels and self.max_disk_usage is None:
            print("Warning: No maximum disk usage specified, using all available space.")
            exit(0)

        if self.max_disk_usage is not None:
            self.max_disk_usage = [int(number) for number in self.max_disk_usage.split(",")]

        self.current_disk_usage = []
        self.max_disk_capacity_reached = []

        for i, cache_folder in enumerate(self.cache_folders):
            # reports in KB
            self.current_disk_usage.append(int(subprocess.check_output(["du", "-sx", cache_folder]).split()[0].decode("utf-8")) * 1e3)
            disk_usage = shutil.disk_usage(cache_folder)

            if i >= len(self.max_disk_usage):  # No maximum specified, therefore use all available space
                self.max_disk_usage.append(self.current_disk_usage[i] + disk_usage.free)

            if self.max_disk_usage[i] - self.current_disk_usage[i] > disk_usage.free:
                print("Warning: maximum allowed disk usage ({} GB) is larger than the available disk space ({} GB).".format(self.max_disk_usage / 1e9, disk_usage.free / 1e9))

            # check if less than 1 MB is free
            self.max_disk_capacity_reached.append({"status": disk_usage.free < 1e6, "timestamp": datetime.datetime.now()})

        if "land" in self.class_names and self.unlabel_chart_data:
            print("Exiting: Land is both declared as a target and filtered out, please adjust configuration.")
            exit(0)

        logging.basicConfig(filename="radar_dataloader.log")
        self.logger = logging.getLogger()

        self.data_loader = DataLoader(self.data_folder, sensor_config=dataloader_config)

        try:
            self.load_files_from_index(osp.join(self.dataset_folder, self.split) + ".txt")
        except IOError as e:
            print(e)
            print("No index file found for dataset, generating index for dataset instead")
            self.update_dataset_file()
            self.redistribute_set_splits(self.split)
            self.load_files_from_index(osp.join(self.dataset_folder, self.split) + ".txt")

            # TODO: calculate mean and other data and save in dataset folder

        try:
            self.read_dataset_stats_from_file()
        except (IOError, MissingStatsError) as e:
            print(e)
            self.mean_bgr = np.array([59.06010940426199])
            #self.calculate_dataset_stats()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_path = self.files[self.split][index]["data"]
        data_range = self.data_ranges[self.files[self.split][index]["range"]]
        label_path = self.files[self.split][index]["label"]

        try:
            img = self.load_image(data_path)
        except DataFileNotFound:
            indexes = list(range(len(self)))
            indexes.remove(index)
            return self.__getitem__(random.choice(indexes))

        # load label
        try:
            lbl = self.get_label(data_path, label_path, data=img, index=index)
        except LabelSourceMissing:
            indexes = list(range(len(self)))
            indexes.remove(index)
            return self.__getitem__(random.choice(indexes))

        if self.coordinate_system == "Cartesian":
            cart = self.data_loader.transform_image_from_sensor(t, sensor, sensor_index, dim=2828, scale=3.39,
                                                               image=np.dstack((img, lbl)).astype(np.int16), use_gpu=True)
            img = cart[:, :, 0].astype(np.uint8)
            lbl = cart[:, :, 1].astype(np.int32)

        img = img[data_range]
        lbl = lbl[data_range]

        if self.include_weather_data:
            img = {"metadata": self.get_weather_data(data_path), "image": img}

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        try:
            img = img.astype(np.float64)
            img -= self.mean_bgr
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).float()
            lbl = torch.from_numpy(lbl).long()
        except Exception as e:
            if isinstance(img, dict):
                img_image = img["image"]
                img_image = img_image.astype(np.float64)
                img_image -= self.mean_bgr
                img_image = np.expand_dims(img_image, axis=0)
                img_image = torch.from_numpy(img_image).float()
                lbl = torch.from_numpy(lbl).long()
                try:
                    img_weather = torch.from_numpy(np.asarray([val for key, val in sorted(img["metadata"].items())])).float()
                except RuntimeError:  # no metadata?
                    self.logger.exception("Runtime error when processing metadata")
                    img_weather = torch.zeros(14).float()
                img["image"] = img_image
                img["metadata"] = img_weather
            else:
                self.logger.exception("Something went wrong when transforming")
                print(e)
                raise e
        return img, lbl

    def untransform(self, img, lbl):
        try:
            img = img.numpy()
            img = np.squeeze(img, axis=0)
            img += self.mean_bgr
            img = img.astype(np.uint8)
            lbl = lbl.numpy()
        except Exception as e:
            if isinstance(img, dict):
                img_image = img["image"].numpy()
                img_image = np.squeeze(img_image, axis=0)
                img_image += self.mean_bgr
                img_image = img_image.astype(np.uint8)
                img_metadata = img["metadata"].numpy()
                img = {"image": img_image, "metadata": img_metadata}
                lbl = lbl.numpy()
        return img, lbl

    def load_image(self, data_path):
        # load image
        basename = osp.splitext(data_path)[0]
        t = self.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

        img = self.data_loader.load_image(t, sensor, sensor_index)
        if type(img) == list:
            raise DataFileNotFound
        return img

    def get_weather_data(self, data_path):
        basename = osp.splitext(data_path)[0]
        t = self.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

        meta_data = self.data_loader.get_metadata(t, sensor, sensor_index)

        try:
            return {key: float(val) for key, val in meta_data["weather"].items()}
        except Exception as e:
            if not meta_data:
                raise MetadataNotFound

            if "weather" not in meta_data:
                raise WeatherDataMissing

            raise e

    def data_path_to_rel_label_path(self, data_path):
        if self.data_folder in data_path:
            rel_label_path = osp.relpath(data_path, start=self.data_folder).replace(".bmp", "_label.npy")
        else:
            rel_label_path = data_path.replace(".bmp", "_label.npy")

        return rel_label_path

    def get_filename(self, index, with_radar=False, with_extension=False):
        if with_radar:
            filename = "/".join(self.files[self.split][index]["data"].split("/")[-2:])
        else:
            filename = self.files[self.split][index]["data"].split("/")[-1]
        return filename if with_extension else osp.splitext(filename)[0]

    def get_data_path(self, radar_relative_path):
        path_split = radar_relative_path.split("/")
        if len(path_split) == 1:
            raise AmbiguousDataPath
        elif len(path_split) < 4:
            filename = path_split[-1]
            file_times = filename.split("-")
            file_times[-1] = file_times[-1][:2]
            radar_relative_path = "{}/{}/{}/{}".format("-".join(file_times[:-1]), "-".join(file_times), path_split[-2], filename)
        return osp.join(self.data_folder, radar_relative_path)

    def get_label_path(self, data_path):
        """ Accepts both radar relative path, and full path"""
        rel_label_path = self.data_path_to_rel_label_path(data_path)

        for label_folder in self.cache_folders:
            label_path = osp.join(label_folder, rel_label_path)
            if osp.exists(label_path):
                return label_path

        return None

    def calculate_dataset_stats(self):
        dataset_mean = self.get_mean()
        self.mean_bgr = dataset_mean
        config = ConfigParser()
        config.read(osp.join(self.dataset_folder, "stats.txt"))
        config["Training"]["MeanBackground"] = dataset_mean
        with open(osp.join(self.dataset_folder, "stats.txt"), 'w') as cfgout:
            config.write(cfgout)


    def read_dataset_stats_from_file(self):
        config = ConfigParser()
        with open(osp.join(self.dataset_folder, "stats.txt"), 'r') as stats_cfg:
            config.read_file(stats_cfg)
            self.mean_bgr = config["Training"].getfloat("MeanBackground")
            if self.mean_bgr is None:
                raise MissingStatsError


    def reload_files_from_default_index(self):
        self.load_files_from_index(osp.join(self.dataset_folder, self.split + ".txt"))

    def load_files_from_index(self, index):
        with open(index, "r+") as file:
            lines = file.readlines()
            file_edited = False
            for line_num, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="Reading dataset index file"):
                line_edited = False
                line = line.strip()

                try:
                    filename, target_locations_string = line.split(";")
                except:
                    print("Invalid format for line number {} in index file. Should be:".format(line_num))
                    print("path/to/file;[target0_0,target0_1]/[target_1_0,target_1_1]/...")
                    print("but is: {}".format(line))
                    exit(0)

                if False:
                    line = "removed"
                    line_edited = True
                else:
                    target_locations = self.ais_targets_string_to_list(target_locations_string)

                    for i, data_range in enumerate(self.data_ranges):
                        if target_locations is None:  # ais target data missing, TODO: actually have to check if targets are hidden...
                            ais_targets = self.get_ais_targets(filename)
                            ais_targets_string = self.ais_targets_to_string(ais_targets)
                            target_locations = self.ais_targets_string_to_list(ais_targets_string)

                            line_edited = True
                            edit_pos = line.rfind(";")

                            line = line[:edit_pos + 1] + ais_targets_string

                        if not self.remove_files_without_targets or any(self.point_in_range(target, data_range, margin=30) for target in target_locations):
                            self.files[self.split].append({ "data": self.get_data_path(filename),
                                                            "label": self.get_label_path(filename) if self.cache_labels else None,
                                                            "range": i
                                                            })
                if line_edited:
                    file_edited = True
                    lines[line_num] = line + "\n"

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

    def update_dataset_file(self, from_time=None, to_time=None):
        def collect_data_files_recursively(parent, filenames=None):
            if filenames is None:
                filenames = []
                to_search = tqdm.tqdm(listdir(parent), desc="Searching for data files", total=len(listdir(parent)), leave=False)
            else:
                to_search = listdir(parent)

            for child in to_search:
                if re.match("^[0-9-]*$", child):
                    if from_time is None and to_time is None or len(child) == 10:
                        filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                    else:
                        child_datetime = datetime.datetime.strptime(child, "%Y-%m-%d{}".format("-%H" if len(child) > 10 else ""))
                        if from_time is not None and to_time is not None and from_time <= child_datetime <= to_time:
                            filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                        elif from_time is None and child_datetime <= to_time or to_time is None and child_datetime >= from_time:
                            filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                elif child in self.radar_types:
                    filenames = collect_data_files_recursively(osp.join(parent, child), filenames)
                elif child.endswith(".bmp"):
                    filenames.append(osp.join(parent, child))

            return filenames

        splits = ["train", "valid", "test"]
        splits.remove(self.split)

        dataset_files = [file["data"] for file in self.files[self.split]]

        for split in splits:
            split_index = osp.join(self.dataset_folder, split + ".txt")
            if osp.exists(split_index):
                with open(split_index, 'r') as index:
                    for line in index.readlines():
                        dataset_files.append(line.strip().split(";")[0])

        dataset_files = set(dataset_files)

        old_cache_labels = self.cache_labels

        files = collect_data_files_recursively(self.data_folder)
        print("Found {} data files".format(len(files)))

        files = [osp.relpath(file, start=self.data_folder) for file in files if file not in dataset_files]

        sorted_files = sorted(files, key=lambda x: datetime.datetime.strptime(x.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S"))

        filter_stats = {"Time": 0, "No targets": 0, "Missing data": 0}

        with open(osp.join(self.root, "processed_files.txt"), "a+") as processed_files_index:
            lines = processed_files_index.readlines()
            files_without_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] in ["false\n", "[]\n"]]
            files_with_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] not in ["false\n", "[]\n"]]

            #last_processed = lines[-1].rstrip("\n").split(";")[0].replace(".bmp", "")
            #last_processed_time = datetime.datetime.strptime(last_processed, "%Y-%m-%d-%H_%M_%S")
            last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_types}
            with open(osp.join(self.dataset_folder, self.split + ".txt"), "a") as index:
                for file in tqdm.tqdm(sorted_files, total=len(sorted_files), desc="Filtering data files", leave=False):
                    try:
                        if self.skip_processed_files and (file in files_without_targets or file in files_with_targets):
                            continue

                        file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                        file_radar_type = file.split("/")[-2]

                        if file_time - last_time[file_radar_type] < datetime.timedelta(seconds=self.min_data_interval):
                            filter_stats["Time"] += 1
                            continue

                        ais_targets = self.get_ais_targets(file)

                        if self.remove_files_without_targets and file not in files_with_targets:
                            if file in files_without_targets:
                                continue
                            ais_targets = self.ais_targets_to_list(ais_targets)
                            ais_targets_in_range = [t for t in ais_targets if t[1] <= self.image_width]

                            if len(ais_targets_in_range) > 0:
                                if self.remove_hidden_targets:
                                    lbl_path = self.get_label_path(file)
                                    try:
                                        lbl = self.get_label(file, throw_save_exception=True)
                                    except MaxDiskUsageError:
                                        print("Allowed disk space used up, terminating search.")
                                        self.cache_labels = False
                                        break
                                    except OSError:
                                        print("No more available disk space, terminating search.")
                                        self.cache_labels = False
                                        break
                                    except LabelSourceMissing:
                                        processed_files_index.write(
                                            "{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                                        continue
                                    if np.any(lbl[:, :self.image_width] == self.LABELS["vessel"]):
                                        processed_files_index.write(
                                            "{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                                    else:
                                        processed_files_index.write(
                                            "{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                                        files_without_targets.append(file)
                                        filter_stats["No targets"] += 1
                                        remove(lbl_path)
                                        continue
                            else:
                                processed_files_index.write("{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                                files_without_targets.append(file)
                                filter_stats["No targets"] += 1
                                continue

                        img = self.load_image(file)
                        if img is None or type(img) == "list":  # check if data is corrupted
                            filter_stats["Missing data"] += 1
                            continue

                        last_time[file_radar_type] = file_time
                        index.write("{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))

                    except Exception as e:  # temporary
                        self.logger.exception("An exception occured while processing images, skipping.\nimage {}".format(file))
                        processed_files_index.write("{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                        files_without_targets.append(file)
                        continue

        self.cache_labels = old_cache_labels

        #print("{} data files left after filtering (time: {}, no targets: {})".format(len(filtered_files), filter_stats["Time"], filter_stats["No targets"]))

    def redistribute_set_splits(self, new_split):
        assert(sum(new_split) == 1)

        files = []
        splits = ["train", "valid", "test"]

        for split in splits:
            split_index = osp.join(self.dataset_folder, split + ".txt")
            if osp.exists(split_index):
                with open(split_index, 'r') as index:
                    for line in index.readlines():
                        files.append(line)

        random.shuffle(files)

        # TODO: maybe sort before writing to index file?

        index_files = [osp.join(self.dataset_folder, split + ".txt") for split in splits]
        with open(index_files[0], 'r+') as train_index, open(index_files[1], 'r+') as valid_index, open(index_files[2], 'r+') as test_index:
            train_index.truncate()
            valid_index.truncate()
            test_index.truncate()
            for i, file in enumerate(files):
                if i <= new_split[0] * len(files):
                    train_index.write(file)
                elif i <= (new_split[1] + new_split[0]) * len(files):
                    valid_index.write(file)
                else:
                    test_index.write(file)

    def get_mean(self):
        mean_sum = 0
        missing_files = []
        processed_files = []

        if "train" not in self.files:
            self.load_files_from_index(osp.join(self.dataset_folder, "train.txt"))

        sorted_files = sorted(self.files["train"],
                              key=lambda x: (datetime.datetime.strptime(x["data"].split("/")[-1].replace(".bmp", ""),
                                                                       "%Y-%m-%d-%H_%M_%S"), x["range"]))
        for i, entry in tqdm.tqdm(enumerate(sorted_files), total=len(sorted_files),
                              desc="Calculating mean for dataset"):
            if i % 5000 == 0 and i != 0:
                print("Mean so far after {} images: {}".format(i, mean_sum/len(processed_files)))
            if entry["data"] in processed_files:
                continue

            # load image
            try:
                img = self.load_image(entry["data"])
            except DataFileNotFound:
                missing_files.append(entry)
                continue

            image_ranges_used = [entry["range"]]
            for j in range(entry["range"] + 1, len(self.data_ranges)):
                if j + i < len(sorted_files):
                    if not entry["data"] == sorted_files[i + j]["data"]:
                        break
                    else:
                        image_ranges_used.append(sorted_files[i + j]["range"])

            if len(image_ranges_used) == len(self.data_ranges):
                mean_sum += np.mean(img[:self.image_height, :self.image_width], dtype=np.float64)
            else:
                image_mean = sum([np.mean(img[self.data_ranges[k]], dtype=np.float64) for k in image_ranges_used])
                mean_sum += image_mean/len(image_ranges_used)
            processed_files.append(entry["data"])

        return mean_sum/len(processed_files)

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

            lbl = self.get_label(file["path"])[:self.image_height, :self.image_width]
            data_range = self.data_ranges[file["range"]]

            for c_index, c in enumerate(self.class_names):
                class_shares[c] += lbl[lbl == c_index].size/lbl.size

        class_shares.update({c: class_shares[c]/len(self.files[self.split]) for c in class_shares.keys()})

        return class_shares

    def save_numpy_file(self, file_rel_path, file, throw_exception=False):
        # TODO: always evalutaes to true for some reason
        #if all([capacity["status"] and datetime.datetime.now() - capacity["timestamp"] < datetime.timedelta(hours=1) for capacity in self.max_disk_capacity_reached]):
        #    if throw_exception:
        #        raise OSError
        #else:
        for i, cache_folder in enumerate(self.cache_folders):
            try:
                if self.current_disk_usage[i] + file.nbytes > self.max_disk_usage[i]:
                    if throw_exception and i == len(self.cache_folders) - 1:
                        raise MaxDiskUsageError
                else:
                    file_path = osp.join(cache_folder, file_rel_path)
                    if not osp.exists(osp.dirname(file_path)):
                        makedirs(osp.dirname(file_path))

                    if not osp.exists(file_path):
                        self.current_disk_usage[i] += file.nbytes

                    np.save(file_path, file)

                    return file_path
            except OSError:  # numpy cannot allocate enough free space
                self.max_disk_capacity_reached[i]["status"] = True
                self.max_disk_capacity_reached[i]["timestamp"] = datetime.datetime.now()
                if throw_exception and i == len(self.cache_folders) - 1:
                    raise OSError

        return None

    def collect_and_cache_labels(self):
        files = self.files[self.split]

        with open(osp.join(self.dataset_folder, "train" if self.split == "valid" else "valid"), "r") as file:
            for line in file:
                line = line.strip()
                files.append(line.split(";"))

        for i, f in enumerate(files):
            print("Caching labels for file {} of {}".format(i, len(files)))
            ais, land = self.get_label(f[0])

    def get_label(self, data_path, label_path=None, data=None, index=None, throw_save_exception=False):
        cached_label_missing = False
        if self.cache_labels:
            try:
                label = np.load(self.get_label_path(data_path)).astype(np.int32)
            except (IOError, AttributeError):
                cached_label_missing = True

        if not self.cache_labels or cached_label_missing:
            label_path = label_path if label_path is not None else self.get_label_path(data_path)

            basename = osp.splitext(data_path)[0]
            t = self.data_loader.get_time_from_basename(basename)
            sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)
            ais = self.data_loader.load_ais_layer_sensor(t, sensor, sensor_index)

            if isinstance(ais, list):
                self.logger.warning("AIS data could not be gathered for {}".format(self.data_path_to_rel_label_path(data_path)))
                raise LabelSourceMissing
            else:
                label = ais.astype(np.int32)[:self.image_height, :self.image_width] * self.LABEL_SOURCE["ais"]

            if not {"land", "islet", "unknown"}.isdisjoint(self.class_names) or self.unlabel_chart_data:
                chart = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index, binary=True, only_first_range_step=True if self.image_width <= 2000 else False)

                if isinstance(chart, list):
                    self.logger.warning("Chart data could not be gathered for {}".format(self.data_path_to_rel_label_path(data_path)))
                    raise LabelSourceMissing
                else:
                    chart = chart[:self.image_height, :self.image_width]

                lbl[chart == 1] = self.LABEL_SOURCE["chart"]

            if cached_label_missing:
                save_path = self.save_numpy_file(self.data_path_to_rel_label_path(data_path), label.astype(np.int8), throw_exception=throw_save_exception)
                if index is not None:
                    self.files[self.split][index]["label"] = save_path

        label = label[:self.image_height, :self.image_width]  # TODO: maybe take in data range?
        # Process label source data
        if "vessel" in self.class_names:
            label[label == self.LABEL_SOURCE["ais"]] = self.LABELS["vessel"]

            if self.min_vessel_land_dist > 0:
                label = cc.remove_vessels_close_to_land(label, distance_threshold=self.min_vessel_land_dist)
        else:
            label[label == self.LABEL_SOURCE["ais"]] = self.LABELS["background"]

        if self.unlabel_chart_data:
            label[label == self.LABEL_SOURCE["chart"]] = self.LABELS["unlabeled"]
        else:
            chart_data = (label == self.LABEL_SOURCE["chart"])
            chart_classified = cc.classify_chart(chart_data,
                                                 classes=[self.LABELS["islet"], self.LABELS["land"]],
                                                 area_threshold=self.chart_area_threshold)

            # for legacy support
            label[(label == self.LABELS["land"]) | (label == self.LABELS["unknown"])] = self.LABELS["background"]

            if self.remove_hidden_targets or "unknown" in self.class_names:
                land = chart_classified == self.LABELS["land"]
                hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
                hidden_by_land_mask[:, 0] = land[:, 0]
                for col in range(1, land.shape[1]):
                    np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])

                if self.remove_hidden_targets:
                    if "unknown" in self.class_names:
                        label[(hidden_by_land_mask == 1) & (chart_classified == 0)] = self.LABELS["unknown"]
                    else:
                        label[(hidden_by_land_mask == 1) & (label == self.LABELS["vessel"])] = self.LABELS["background"]
                else:
                    label[(hidden_by_land_mask == 1) & (chart_classified == 0) & (label != self.LABELS["vessel"])] = self.LABELS["unknown"]


            if "land" in self.class_names or "islet" in self.class_names:
                if data is None:
                    data = self.load_image(data_path)[:label.shape[0], :label.shape[1]]
                else:
                    data = data[:label.shape[0], :label.shape[1]]

                chart_classified[(chart_data == 1) & (data < self.land_threshold)] = 0

                # TODO: possible speed up here with np.where or similar?
                if "islet" in self.class_names and "land" in self.class_names:
                    label = np.where((chart_classified != 0), chart_classified, label)
                elif "islet" in self.class_names:
                    label[chart_classified == self.LABELS["islet"]] = self.LABELS["islet"]
                else:
                    label[chart_classified == self.LABELS["land"]] = self.LABELS["land"]

                if "unknown" in self.class_names:
                    label[(chart_classified == 0) & (chart_data == 1) & (hidden_by_land_mask == 1)] = self.LABELS["unknown"]
                else:
                    label[(chart_classified == 0) & (chart_data == 1)] = self.LABELS["background"]
            else:
                if "unknown" in self.class_names:
                    label[(chart_classified != 0) | (label != self.LABELS["unknown"])] = self.LABELS["background"]
                else:
                    label[(chart_classified != 0) | (label == self.LABELS["unknown"])] = self.LABELS["background"]

        # unlabel data blocked by mast for Radar0
        if "Radar0" in data_path:
            label[2000:2080, :] = self.LABELS["unlabeled"]

        return label

    def update_cached_labels(self, components):
        processed_labels = set()
        for entry in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]), desc="Updating cached labels", leave=False):
            label_path = self.get_label_path(entry["data"])
            if label_path in processed_labels:
                continue

            data_path = entry["data"]
            try:
                label = np.load(label_path).astype(np.int16)
            except IOError as e:
                print("Label does not exists at {}".format(label_path))
                continue

            label = label[:self.image_height, :self.image_width]

            for component in components:
                if component in self.LABELS:
                    label[label == self.LABELS[component]] = self.LABELS["background"]
            if "chart" in components:
                if "Radar0" in data_path:
                    label[(label == self.LABELS["land"]) | (label == self.LABELS["unknown"])] = self.LABELS["background"]
                else:
                    label[label == self.LABELS["unknown"]] = self.LABELS["background"]

            basename = osp.splitext(data_path)[0]
            t = self.data_loader.get_time_from_basename(basename)
            sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

            if "ais" in components:
                ais = self.data_loader.load_ais_layer_sensor(t, sensor, sensor_index)
                if isinstance(ais, list):
                    self.logger.warning("Label {} not updated.\nAIS data could not be gathered".format(label_path))
                    processed_labels.append(label_path)
                    continue
                else:
                    ais = ais[:self.image_height, :self.image_width]
                label[ais == 1] = self.LABEL_SOURCE["ais"]
            if "chart" in components and "Radar0" in data_path:
                land = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index, binary=True, only_first_range_step=True if self.image_width <= 2000 else False)
                if isinstance(land, list):
                    self.logger.warning("Label {} not updated.\nChart data could not be gathered".format(label_path))
                    processed_labels.append(label_path)
                    continue
                else:
                    land = land[:self.image_height, :self.image_width]

                label[land == 1] = self.LABEL_SOURCE["chart"]

            self.save_numpy_file(label_path, label.astype(np.int8))
            processed_labels.add(label_path)

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
                    if "land" in self.class_names or (self.unlabel_chart_data and self.remove_hidden_targets):
                        lines.append(data_file.replace(".bmp", "_label_land.npy\n"))
                    if self.remove_hidden_targets:
                        lines.append(data_file.replace(".bmp", "_label_land_hidden.npy\n"))

                last_filename = data_file
                file.writelines(lines)

    def point_in_range(self, point, data_range, margin=0):
        return ((data_range[0].start is None or point[0] - margin >= data_range[0].start) and (
                    data_range[0].stop is None or point[0] + margin <= data_range[0].stop)) \
               and ((data_range[1].start is None or point[1] - margin >= data_range[1].start) and (
                    data_range[1].stop is None or point[1] + margin <= data_range[1].stop))

    def ais_targets_to_list(self, ais_targets, locations_per_target=1):
        if isinstance(ais_targets, dict):
            ais_targets = [target for target in ais_targets.values()]
        assert type(ais_targets[0]) == np.ndarray
        res = []

        for target in ais_targets:
            if locations_per_target == 1:
                res.append(np.round(target[0], decimals=0).astype(np.int64).tolist())
            else:
                target_locs = []
                for target_loc in target:
                    target_locs.append(np.round(target_loc, decimals=0).astype(np.int64).tolist())
                    if len(target_locs) >= locations_per_target:
                        break
                res.append(target_locs)

        return res

    def ais_targets_to_string(self, ais_targets, locations_per_target=1):
        if len(ais_targets) == 0:
            return "[]"
        if isinstance(ais_targets, dict) or type(ais_targets[0]) == np.ndarray:
            ais_targets = self.ais_targets_to_list(ais_targets, locations_per_target=locations_per_target)

        if locations_per_target != 1:
            raise NotImplementedError

        return "/".join([str(t) for t in ais_targets])

    def ais_targets_string_to_list(self, ais_targets_string):
        ais_targets = []

        if ais_targets_string == "":
            return None
        elif ais_targets_string == "[]":
            return ais_targets

        for target in ais_targets_string.split("/"):
            target = target.strip("[]").split(",")
            ais_targets.append([int(target[0]), int(target[1])])

        return ais_targets

    def get_ais_targets(self, data_path):
        basename = osp.splitext(data_path)[0]
        t = self.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

        ais_targets = self.data_loader.load_ais_targets_sensor(t, sensor, sensor_index)
        if isinstance(ais_targets, tuple):
            ais_targets = ais_targets[1]

        return ais_targets

    def set_data_ranges(self, height_division_count, width_division_count, overlap=0):
        if height_division_count < 0 or width_division_count < 0 or overlap < 0:
            raise ValueError

        try:
            height_division_count = int(height_division_count)
            width_division_count = int(width_division_count)
            overlap = int(overlap)
        except:
            print("All input arguments must be positive integers")
            raise ValueError

        self.data_ranges = []
        h_step_size = int(self.image_height / (height_division_count + 1))
        w_step_size = int(self.image_width / (width_division_count + 1))

        for i in range(height_division_count + 1):
            for j in range(width_division_count + 1):
                self.data_ranges.append(
                    np.s_[
                        h_step_size * i - (overlap if i != 0 else 0): h_step_size * (i+1) + (overlap if i != height_division_count else 0),
                        w_step_size * j - (overlap if j != 0 else 0): w_step_size * (j+1) + (overlap if j != width_division_count else 0)
                    ]
                )

    def show_image(self, index=None, data_path=None):
        if index is None and data_path is None:
            return
        if index is not None:
            data_path = self.files[self.split][index]["data"]
            data_range = self.data_ranges[self.files[self.split][index]["range"]]
        else:
            data_path = self.get_data_path(data_path)
            data_range = np.s_[:, :]

        img = self.load_image(data_path)[data_range]
        plt.imshow(img)
        plt.show()

    def show_label(self, index=None, label=None):
        if index is None and label is None:
            return

        if label is None:
            data_path = self.files[self.split][index]["data"]
            data_range = self.data_ranges[self.files[self.split][index]["range"]]
            label_path = self.files[self.split][index]["label"]
            label = self.get_label(data_path, label_path, index=index)[data_range]
        plt.imshow(label)
        plt.colorbar()
        plt.show()

    def show_image_with_label(self, index=None, data_path=None):
        if index is None and data_path is None:
            return

        if index is not None:
            data_path = self.files[self.split][index]["data"]
            data_range = self.data_ranges[self.files[self.split][index]["range"]]
            label_path = self.files[self.split][index]["label"]
        else:
            data_path = self.get_data_path(data_path)
            label_path = self.get_label_path(data_path)
            data_range = np.s_[:self.image_height, :self.image_width]

        img = self.load_image(data_path)[data_range]
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, 2)

        lbl = self.get_label(data_path, label_path, index=index)[data_range]

        print(np.transpose(np.where(lbl == 1)))
        res = label2rgb(lbl, img, n_labels=len(self.class_names))
        plt.imshow(res)
        plt.show()


    def _colorbar(self, mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)


if __name__ == "__main__":
    from polarlys.dataloader import DataLoader

    #np.s_[:int(4096/3), 0:2000], np.s_[int(4096/3):int(2*4096/3), 0:2000], np.s_[int(2*4096/3):, 0:2000]
    dataset = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets", cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt", split="train", dataset_name="test")

    if False:
        if False:
            files = dataset.files[dataset.split]
            new_files = []

            from_time = datetime.datetime(year=2017, month=10, day=28, hour=4, minute=0, second=0)
            to_time = datetime.datetime(year=2018, month=10, day=28, hour=7, minute=38, second=53)
            for file in files:
                file_time = datetime.datetime.strptime(file["data"].split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                if from_time <= file_time <= to_time:
                    new_files.append(file)

            new_files = sorted(new_files, key=lambda x: datetime.datetime.strptime(x["data"].split("/")[-1].replace(".bmp", ""),
                                                                                  "%Y-%m-%d-%H_%M_%S"))
            dataset.files[dataset.split] = new_files
            print("yay")
        if False:
            dataset.files["train"] = dataset.files["train"][92693:]
        dataset.update_cached_labels(("ais", "chart", "unlabeled"))
        #dataset.update_dataset_file(from_time=datetime.datetime(year=2017, month=11, day=27, hour=20, minute=0))
    if True:
        basename = osp.splitext(dataset.files["train"][0]["data"])[0]
        t = dataset.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = dataset.data_loader.get_sensor_from_basename(basename)

        ais_targets = dataset.data_loader.load_ais_targets_sensor(t, sensor, sensor_index)


        dataset_collect = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                             cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt",
                                             split="train", dataset_name="lacie")
        dataset_collect.update_dataset_file(from_time=datetime.datetime(year=2017, month=11, day=30, hour=7, minute=38))
    elif False:
        print(dataset.get_mean())
    elif False:
        files_to_update = []
        processed_files = set()
        for file in tqdm.tqdm(dataset.files["train"], total=len(dataset.files["train"])):
            if file["data"] in processed_files:
                continue
            lbl = np.load(file["label"])
            if np.any((lbl == 3) | (lbl == -1)):
                files_to_update.append(file)
            processed_files.add(file["data"])
        dataset.files["train"] = files_to_update
        dataset.update_cached_labels(("ais", "chart", "unlabeled"))
    elif False:
        for i in range(len(dataset.files["train"])):
            dataset.show_image_with_label(i)
        img, lbl = dataset[0]

    #dataset.redistribute_set_splits([0.97, 0.03, 0])
    #dataset.show_image_with_label(5000)
    exit(0)
    print("mean: ")
    print(dataset.get_mean())
    mean_cols = dataset.get_mean_of_columns()
    np.savetxt("column_sum_new.txt", mean_cols)

    print("class shares: ")
    class_shares = dataset.get_class_shares()
    print(class_shares)
    print("hei")
    #dataset.generate_list_of_required_files()
    #print("hei")
    #for i in tqdm.tqdm(range(len(dataset.files[dataset.split])), total=len(dataset.files[dataset.split])):
    #    img, data = dataset[i]

    #print("Training set mean: {}".format(train.get_mean()))
    #print("Training set class shares: {}".format(train.get_class_shares()))
    #test = np.load("/media/stx/LaCie1/export/2017-10-25/2017-10-25-17/Radar0/2017-10-25-17_00_02_513_label_land.npy")
    #print(test)
else:
    from polarlys.dataloader import DataLoader


