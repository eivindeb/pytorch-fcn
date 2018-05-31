#!/usr/bin/env python

import collections
import os.path as osp
import json

import numpy as np
import PIL.Image
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
import copy
import math
import scipy


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

    def __init__(self, root, dataset_name, cfg, split='train', transform=False, **kwargs):
        self.root = root
        self.split = split
        self._transform = transform
        self.files = collections.defaultdict(list)
        self.dataset_folder = osp.join(self.root, dataset_name)
        if not osp.exists(self.dataset_folder):
            makedirs(self.dataset_folder)

        self.config = ConfigParser()
        self.config.optionxform = str
        try:
            with open(cfg, 'r') as cfg:
                self.config.read_file(cfg)
        except IOError:
            print("Configuration file not found at {}".format(cfg))
            exit(0)
        self.cache_labels = self.config["Parameters"].getboolean("CacheLabels", False)
        self.data_folder = self.config["Paths"].get("DataFolder")
        self.cache_folders = self.config["Paths"].get("CacheFolder(s)").split(",")

        if self.cache_labels and self.cache_folders is None:
            print("Cache labels is set to true, but no label folder is provided.")
            exit(0)

        for cache_folder in self.cache_folders:
            if not osp.exists(cache_folder):
                makedirs(cache_folder)

        dataloader_config = self.config["Paths"].get("DataloaderCFG")

        if dataloader_config is None:
            print("Configuration file missing required field: DataloaderCFG")
            exit(0)

        if self.data_folder is None:
            print("Configuration file missing required field: DataFolder")
            exit(0)

        self.radar_types = [radar for radar in self.config["Parameters"].get("RadarTypes", "Radar0,Radar1,Radar2").split(",")]
        self.unlabel_chart_data = self.config["Parameters"].getboolean("UnlabelChartData", False)
        self.remove_hidden_targets = self.config["Parameters"].getboolean("RemoveHiddenTargets", True)
        self.class_names = np.array([c for c in self.config["Parameters"].get("Classes", "background,ship").split(",")])
        self.class_weights = np.array([int(weight) for weight in self.config["Parameters"].get("ClassWeights", "1,5000").split(",")])
        self.remove_files_without_targets = self.config["Parameters"].getboolean("RemoveFilesWithoutTargets", True)
        self.min_data_interval = self.config["Parameters"].getint("MinDataIntervalSeconds", 0)
        self.skip_processed_files = self.config["Parameters"].getboolean("SkipProcessedFiles", True)
        self.coordinate_system = self.config["Parameters"].get("CoordinateSystem", "Polar")
        self.max_disk_usage = self.config["Parameters"].get("MaximumDiskUsage", None)
        self.set_splits = [float(s) for s in self.config["Parameters"].get("SetSplits", "0.95,0.025,0.025").split(",")]
        self.width_region = [int(s) for s in self.config["Parameters"].get("WidthRegion", "0,2000").split(",")]
        self.height_region = [int(s) for s in self.config["Parameters"].get("HeightRegion", "0,4096").split(",")]
        self.land_threshold = self.config["Parameters"].getint("LandThreshold", 70)
        self.include_weather_data = self.config["Parameters"].getboolean("IncludeWeatherData", False)
        self.chart_area_threshold = self.config["Parameters"].getint("ChartAreaThreshold", 10000)
        self.min_vessel_land_dist = self.config["Parameters"].getfloat("MinVesselLandDistance", 10)
        self.min_own_velocity = self.config["Parameters"].getfloat("MinOwnVelocity", 1)
        self.downsampling_factor = self.config["Parameters"].getfloat("DownsamplingFactor", 1)
        self.image_mode = self.config["Parameters"].get("ImageMode", "Grayscale")
        self.range_normalize = self.config["Parameters"].getboolean("RangeNormalize", False)

        self.height_divisions = self.config["Parameters"].getint("HeightDivisions", 2)
        self.width_divisions = self.config["Parameters"].getint("WidthDivisions", 0)
        self.overlap = self.config["Parameters"].getint("Overlap", 20)

        for var, val in kwargs.items():
            if hasattr(self, var):
                setattr(self, var, val)
                config_var = [var_split.capitalize() for var_split in var.split("_")]
                config_var = "".join(config_var)
                self.config["Parameters"][config_var] = str(val)
            else:
                raise TypeError("Unrecognized keyword argument: {}".format(var))

        if self.class_names.size != self.class_weights.size:
            print("Number of defined classes and class weights does not match.")
            exit(0)

        self.metadata = True if self.include_weather_data else False

        if self.metadata:
            self.metadata_stats = dict()

        self.set_data_ranges(self.height_divisions, self.width_divisions, overlap=self.overlap)

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
            if type(e) is IOError:
                self.calculate_dataset_stats(mode="data")
                if self.metadata:
                    self.calculate_dataset_stats(mode="metadata")
            else:
                self.mean_bgr = np.array([59.06010940426199])  # temporary
                if self.metadata:
                    raise e

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

        if self.downsampling_factor > 1:
            img = cv2.resize(img, None, fx=1 / self.downsampling_factor, fy=1 / self.downsampling_factor, interpolation=cv2.INTER_AREA)

        # load label
        try:
            lbl = self.get_label(data_path, label_path, data=img, index=index)
        except LabelSourceMissing:
            indexes = list(range(len(self)))
            indexes.remove(index)
            return self.__getitem__(random.choice(indexes))
        try:
            if self.coordinate_system == "Cartesian":
                cart_path = osp.join("/mnt/lacie/cartesian/", osp.relpath(data_path, start=self.data_folder)).replace(".bmp", "")
                try:
                    #raise ValueError
                    cart = np.load("{}.npy".format(cart_path))
                except:
                    #raise ValueError
                    basename = osp.splitext(data_path)[0]
                    t = self.data_loader.get_time_from_basename(basename)
                    sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)
                    #img = img[:lbl.shape[0], :lbl.shape[1]]
                    lbl = np.pad(lbl, ((0, 0), (0, 1400)), mode="constant")
                    lbl[lbl == -1] = 0
                    cart = self.data_loader.transform_image_from_sensor(t, sensor, sensor_index, dim=2828, scale_in=1, scale_out=3.39,
                                                                       image=np.dstack((img, lbl)).astype(np.int16), use_gpu=True)

                    if not osp.exists(osp.dirname(cart_path)):
                        makedirs(osp.dirname(cart_path))

                    np.save(cart_path, cart.astype(np.int8))
                img = cart[:, :, 0].astype(np.uint8)
                lbl = cart[:, :, 1].astype(np.int32)
        except:
            indexes = list(range(len(self)))
            indexes.remove(index)
            return self.__getitem__(random.choice(indexes))

        img = img[data_range]
        lbl = lbl[data_range]
        try:
            if self._transform:
                if self.include_weather_data:
                    return self.transform(img, lbl, metadata=self.get_weather_data(data_path))
                else:
                    return self.transform(img, lbl)
            else:
                if self.include_weather_data:
                    return img, lbl, self.get_weather_data(data_path)
                else:
                    return img, lbl
        except:
            indexes = list(range(len(self)))
            indexes.remove(index)
            return self.__getitem__(random.choice(indexes))

    def transform(self, img, lbl, metadata=None):
        img = img.astype(np.float64)
        if self.range_normalize:
            img = self.apply_range_normalization(img)

        img -= self.mean_bgr
        if self.image_mode == "Grayscale":
            img = np.expand_dims(img, axis=0)
        else:
            img = np.stack((img,) * 3, -1)
            img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        if metadata is not None:
            try:
                met_vals = []
                new_max, new_min = 1, -1
                for key, val in sorted(metadata.items()):
                    if "[deg]" in key:
                        val = math.cos(val)
                    old_range = self.metadata_stats[key]["max"] - self.metadata_stats[key]["min"]
                    new_range = new_max - new_min
                    met_vals.append((((val - self.metadata_stats[key]["min"]) * new_range) / old_range) + new_min)
                metadata = torch.from_numpy(np.asarray(met_vals)).float()
            except RuntimeError:  # no metadata?
                self.logger.exception("Runtime error when processing metadata")
                metadata = torch.zeros(14).float()
            return img, lbl, metadata
        else:
            return img, lbl

    def untransform(self, img, lbl, metadata=None):
        img = img.numpy()
        if self.image_mode == "Grayscale":
            img = np.squeeze(img, axis=0)
        else:
            img = img[0, :, :]
        img += self.mean_bgr
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        if metadata is not None:
            metadata = metadata.numpy()
            return img, lbl, metadata
        else:
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

    def apply_range_normalization(self, data):
        """
        #means = [83.68624262883348,67.55744366227191,46.75703295544134]
        means = np.load("/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/torchfcn/datasets/mean_sum.npy")
        stds = np.load("/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/torchfcn/datasets/std_sum.npy")
        #stds = [41.751658620945435,41.92119308299246,39.04992628116686]
        #ranges = [[0, 250], [250, 750], [750, 2000]]
        data -= means
        data /= stds
        return data
        """
        bgr_image = np.load("/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/torchfcn/datasets/mean_bgr_filled.npy")
        bgr_image = bgr_image[:data.shape[1]]
        bgr_image = np.tile(bgr_image, (data.shape[0], 1))
        dived = data / bgr_image
        normalization_constant = np.mean(data) / (np.mean(data / bgr_image))
        data = dived * normalization_constant
        data = (255 / data.max()) * data

        return data

    def homomorphic_filtering(self, data_path=None, img=None):
        if data_path is None and img is None:
            raise ValueError

        if img is None:
            img = self.load_image(data_path)[:1000, 25:2000]

        # Number of rows and columns
        rows = img.shape[0]
        cols = img.shape[1]

        # Convert image to 0 to 1, then do log(1 + I)
        imgLog = np.log1p(np.array(img, dtype="float") / 255)

        # Create Gaussian mask of sigma = 10
        M = 2 * rows + 1
        N = 2 * cols + 1
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
        centerX = np.ceil(N / 2)
        centerY = np.ceil(M / 2)
        gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

        # Low pass and high pass filters
        Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
        Hhigh = 1 - Hlow

        # Move origin of filters so that it's at the top left corner to
        # match with the input image
        HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
        HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

        # Filter the image and crop
        If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
        Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
        Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

        # Set scaling factors and add
        gamma1 = 0.3
        gamma2 = 1.5
        Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

        # Anti-log then rescale to [0,1]
        Ihmf = np.expm1(Iout)
        Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        return np.array(255 * Ihmf, dtype="uint8")

    def get_metadata(self, data_path):
        basename = osp.splitext(data_path)[0]
        t = self.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)

        meta_data = self.data_loader.get_metadata(t, sensor, sensor_index)

        if not meta_data:
            raise MetadataNotFound("Metadata not found for {}".format(data_path))

        return meta_data

    def get_own_vessel_velocity(self, data_path, vector=False):
        vel = self.get_metadata(data_path)["own_vessel_end"]["velocity"]
        if vector:
            return vel

        return math.sqrt(sum(v ** 2 for v in vel))

    def get_weather_data(self, data_path):
        meta_data = self.get_metadata(data_path)

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

    def dump_config(self, path):
        with open(path, 'w') as cfgout:
            self.config.write(cfgout)


    def calculate_dataset_stats(self, mode="data"):
        config = ConfigParser()
        config.optionxform = str
        config.read(osp.join(self.dataset_folder, "stats.txt"))
        if mode == "data":
            if "Training" not in config.sections():
                config["Training"] = {}
            dataset_mean = self.get_mean()
            self.mean_bgr = dataset_mean
            config["Training"]["MeanBackground"] = dataset_mean
        elif mode == "metadata":
            if "Metadata" not in config.sections():
                config["Metadata"] = {}
            self.metadata_stats = self.get_metadata_stats()
            for key, val in self.metadata_stats.items():
                config["Metadata"][key] = "{},{},{}".format(val["min"], val["max"], val["mean"])
        with open(osp.join(self.dataset_folder, "stats.txt"), 'w') as cfgout:
            config.write(cfgout)


    def read_dataset_stats_from_file(self):
        config = ConfigParser()
        config.optionxform = str
        with open(osp.join(self.dataset_folder, "stats.txt"), 'r') as stats_cfg:
            config.read_file(stats_cfg)
            bgr_section = "Training" if not self.range_normalize else "RangeNormalize"
            self.mean_bgr = config[bgr_section].getfloat("MeanBackground")
            if self.mean_bgr is None:
                self.calculate_dataset_stats(mode="data")

            if self.metadata:
                if "Metadata" not in config.sections():
                    self.calculate_dataset_stats(mode="metadata")
                else:
                    try:
                        for key, val in config["Metadata"].items():
                            val_min, val_max, val_mean = val.split(",")
                            val_min, val_max, val_mean = float(val_min), float(val_max), float(val_mean)
                            self.metadata_stats.update({key: {"min": val_min, "max": val_max, "mean": val_mean}})
                    except:
                        raise MissingStatsError("Metadata statistics are wrongly formatted, should be: key = min, max, mean\n is: {} = {}".format(key, val))



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
                    
                    if target_locations is None:  # ais target data missing, TODO: actually have to check if targets are hidden...
                        ais_targets = self.get_ais_targets(filename)
                        ais_targets_string = self.ais_targets_to_string(ais_targets)
                        target_locations = self.ais_targets_string_to_list(ais_targets_string)

                        line_edited = True
                        edit_pos = line.rfind(";")

                        line = line[:edit_pos + 1] + ais_targets_string

                    target_locations = [[loc[0] / self.downsampling_factor, loc[1] / self.downsampling_factor] for loc in target_locations]

                    for i, data_range in enumerate(self.data_ranges):


                        if not self.remove_files_without_targets or any(self.point_in_range(target, data_range, margin=30 / self.downsampling_factor) for target in target_locations):
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

        self.sort_chronologically()

        if self.min_data_interval > 0:
            last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_types}

            new_files = []

            for file_info in self.files[self.split]:
                file = file_info["data"]
                file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                file_radar_type = file.split("/")[-2]

                if file_time == last_time[file_radar_type]:  # different range of same image
                    new_files.append(file_info)
                    continue

                if file_time - last_time[file_radar_type] > datetime.timedelta(seconds=self.min_data_interval):
                    last_time[file_radar_type] = file_time
                    new_files.append(file_info)

            self.files[self.split] = new_files


    def collect_data_files_recursively(self, parent, from_time=None, to_time=None):
        if from_time is not None:
            from_time_hour = datetime.datetime(year=from_time.year, month=from_time.month, day=from_time.day,
                                               hour=from_time.hour)
        if to_time is not None:
            to_time_hour = datetime.datetime(year=to_time.year, month=to_time.month, day=to_time.day, hour=to_time.hour)

        for child in sorted(listdir(parent)):
            if re.match("^[0-9-]*$", child):
                if from_time is None and to_time is None or len(child) == 10:
                    yield from self.collect_data_files_recursively(osp.join(parent, child), from_time, to_time)
                else:
                    child_datetime = datetime.datetime.strptime(child,
                                                                "%Y-%m-%d{}".format("-%H" if len(child) > 10 else ""))
                    if from_time is not None and to_time is not None and from_time_hour <= child_datetime <= to_time_hour:
                        yield from self.collect_data_files_recursively(osp.join(parent, child), from_time, to_time)
                    elif from_time is None and child_datetime <= to_time_hour or to_time is None and child_datetime >= from_time_hour:
                        yield from self.collect_data_files_recursively(osp.join(parent, child), from_time, to_time)
            elif child in self.radar_types:
                yield from self.collect_data_files_recursively(osp.join(parent, child), from_time, to_time)
            elif child.endswith(".bmp"):
                if from_time is None and to_time is None:
                    yield osp.join(parent, child)
                else:
                    child_datetime = datetime.datetime.strptime(child.replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                    if (from_time is None and to_time >= child_datetime) or (
                            to_time is None and from_time <= child_datetime):
                        yield osp.join(parent, child)
                    elif from_time <= child_datetime <= to_time:
                        yield osp.join(parent, child)

    def find_streak_candidates(self, min_streak_length=120, from_time=None, to_time=None):
        streak_start = None
        streak_vessels = None
        with open(osp.join(self.root, "streak_candidates.txt"), "a+") as index:
            for file in self.collect_data_files_recursively(self.data_folder, from_time=from_time, to_time=to_time):
                if self.min_own_velocity > 0 and self.get_own_vessel_velocity(file) < self.min_own_velocity:
                    continue

                ais_targets = self.get_ais_targets(file)

                if len(ais_targets) == 0:
                    continue

                file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")

                ais_targets = self.ais_targets_to_list(ais_targets)
                ais_targets_in_range = [t for t in ais_targets if self.width_region[0] <= t[1] <= self.width_region[1]]

                if len(ais_targets_in_range) > 0:
                    if streak_start is None:
                        streak_start = file_time
                        streak_vessels = []

                    streak_vessels.append(len(ais_targets_in_range))
                elif streak_start is not None:
                    streak_length = abs(file_time - streak_start)
                    if streak_length >= datetime.timedelta(seconds=min_streak_length):
                        streak_string = "{}/{}/{}\n".format(file, streak_length, sum(streak_vessels) / len(streak_vessels))
                        print("Added new streak:\n{}".format(streak_string))
                        index.write(streak_string)
                        index.flush()
                    streak_start = None






    def update_dataset_file(self, from_time=None, to_time=None):
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

        filter_stats = {"Time": 0, "No targets": 0, "Missing data": 0, "Velocity": 0}

        with open(osp.join(self.root, "processed_files.txt"), "a+") as processed_files_index:
            lines = processed_files_index.readlines()
            files_without_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] in ["false\n", "[]\n"]]
            files_with_targets = [f.rstrip("\n") for f in lines if f.split(";")[1] not in ["false\n", "[]\n"]]

            #last_processed = lines[-1].rstrip("\n").split(";")[0].replace(".bmp", "")
            #last_processed_time = datetime.datetime.strptime(last_processed, "%Y-%m-%d-%H_%M_%S")
            last_time = {radar_type: datetime.datetime(year=2000, month=1, day=1) for radar_type in self.radar_types}
            with open(osp.join(self.dataset_folder, self.split + ".txt"), "a") as index:
                for file in tqdm.tqdm(self.collect_data_files_recursively(self.data_folder, from_time, to_time), desc="Filtering data files", leave=False):
                    try:
                        if self.skip_processed_files and (file in files_without_targets or file in files_with_targets):
                            continue

                        if file in dataset_files:
                            continue

                        file_time = datetime.datetime.strptime(file.split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
                        file_radar_type = file.split("/")[-2]

                        if file_time - last_time[file_radar_type] < datetime.timedelta(seconds=self.min_data_interval):
                            filter_stats["Time"] += 1
                            continue

                        if self.min_own_velocity > 0 and self.get_own_vessel_velocity(file) < self.min_own_velocity:
                            filter_stats["Velocity"] += 1
                            continue

                        ais_targets = self.get_ais_targets(file)

                        if self.remove_files_without_targets and file not in files_with_targets:
                            if file in files_without_targets:
                                continue
                            ais_targets = self.ais_targets_to_list(ais_targets)
                            ais_targets_in_range = [t for t in ais_targets if self.width_region[0] <= t[1] <= self.width_region[1]]

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
                                    if np.any(lbl[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]] == self.LABELS["vessel"]):
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
                        index.write("{};{}\n".format(osp.relpath(file, start=self.data_folder), self.ais_targets_to_string(ais_targets)))
                        index.flush()
                        print("Added {} to dataset".format(file))

                    except Exception as e:  # temporary
                        self.logger.exception("An exception occured while processing images, skipping.\nimage {}".format(file))
                        processed_files_index.write("{};{}\n".format(file, self.ais_targets_to_string(ais_targets)))
                        files_without_targets.append(file)
                        continue

        self.cache_labels = old_cache_labels

        #print("{} data files left after filtering (time: {}, no targets: {})".format(len(filtered_files), filter_stats["Time"], filter_stats["No targets"]))

    def shuffle(self, seed):
        self.sort_chronologically(split=self.split)

        random.Random(seed).shuffle(self.files[self.split])

    def sort_chronologically(self, split=None):
        if split is None:
            split = self.split

        self.files[self.split] = sorted(self.files[self.split], key=lambda x: (datetime.datetime.strptime(
            x["data"].split("/")[-1].replace(".bmp", ""),
            "%Y-%m-%d-%H_%M_%S"), x["range"]))

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

    def get_metadata_stats(self):
        if "train" not in self.files:
            self.load_files_from_index(osp.join(self.dataset_folder, "train.txt"))

        metadata_stats = {}
        processed_files = set()

        for i, entry in tqdm.tqdm(enumerate(self.files["train"]), total=len(self.files["train"]),
                              desc="Calculating statistics for metadata"):
            if entry["data"] in processed_files:
                continue

            try:
                metadata = self.get_weather_data(entry["data"])
            except MetadataNotFound:
                continue
            for key, val in metadata.items():
                if "[deg]" in key:
                    val = math.cos(val)
                if key not in metadata_stats:
                    metadata_stats.update({key: {"min": val, "max": val, "mean": val}})
                else:
                    if val < metadata_stats[key]["min"]:
                        metadata_stats[key]["min"] = val
                    if val > metadata_stats[key]["max"]:
                        metadata_stats[key]["max"] = val
                    metadata_stats[key]["mean"] += val

            processed_files.add(entry["data"])
                    
        for val in metadata_stats.values():
            val["mean"] = val["mean"] / len(processed_files)

        return metadata_stats

    def get_mean(self):
        mean_sum = 0
        missing_files = []
        processed_files = []

        for i, entry in tqdm.tqdm(enumerate(self.files["train"]), total=len(self.files["train"]),
                              desc="Calculating mean for dataset"):
            if i % 100 == 0 and i != 0:
                print("Mean so far after {} images: {}".format(i, mean_sum/len(processed_files)))
            if entry["data"] in processed_files:
                continue

            # load image
            try:
                img = self.load_image(entry["data"])
            except DataFileNotFound:
                missing_files.append(entry)
                continue

            img = img[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]
            if self.range_normalize:
                img = self.apply_range_normalization(img)

            mean_sum += np.mean(img, dtype=np.float64)
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

            lbl = self.get_label(file["path"])[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]
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

            if ais is None:
                self.logger.warning("AIS data could not be gathered for {}".format(self.data_path_to_rel_label_path(data_path)))
                raise LabelSourceMissing
            else:
                label = ais.astype(np.int32)[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]] * self.LABEL_SOURCE["ais"]

            if not {"land", "islet", "unknown"}.isdisjoint(self.class_names) or self.unlabel_chart_data:
                chart = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index, binary=True, only_first_range_step=True if self.width_region[1] <= 2000 else False)

                if chart is None:
                    self.logger.warning("Chart data could not be gathered for {}".format(self.data_path_to_rel_label_path(data_path)))
                    raise LabelSourceMissing
                else:
                    chart = chart[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]

                label[chart == 1] = self.LABEL_SOURCE["chart"]

            if cached_label_missing:
                save_path = self.save_numpy_file(self.data_path_to_rel_label_path(data_path), label.astype(np.int8), throw_exception=throw_save_exception)
                if index is not None:
                    self.files[self.split][index]["label"] = save_path

        label = label[:4096, :2000]  # TODO: maybe take in data range?

        if self.downsampling_factor > 1:
            label[label == self.LABEL_SOURCE["ais"]] = self.LABELS["background"]
            label = cv2.resize(label.astype(np.int16), None, fx=1 / self.downsampling_factor, fy=1 / self.downsampling_factor, interpolation=cv2.INTER_AREA)
            label = label.astype(np.int32)  # cv2.resize does not work with int32 for some reason

            basename = osp.splitext(data_path)[0]
            t = self.data_loader.get_time_from_basename(basename)
            sensor, sensor_index = self.data_loader.get_sensor_from_basename(basename)
            scaled_ais_layer = self.data_loader.load_ais_layer_sensor(t, sensor, sensor_index, scale= 1 / self.downsampling_factor, binary=True)
            if scaled_ais_layer is None:
                self.logger.warning("Scaled AIS data could not be gathered for {}".format(self.data_path_to_rel_label_path(data_path)))
                raise LabelSourceMissing

            #scaled_range = [scaled_ais_layer.shape[0] - round((scaled_ais_layer.shape[0] * self.downsampling_factor - self.image_height) / self.downsampling_factor),
            #                scaled_ais_layer.shape[1] - round((scaled_ais_layer.shape[1] * self.downsampling_factor - self.image_width) / self.downsampling_factor)]
            scaled_ais_layer = scaled_ais_layer[:label.shape[0], :label.shape[1]]

            if scaled_ais_layer.shape[0] < label.shape[0]:
                scaled_ais_layer = np.pad(scaled_ais_layer, ((0, label.shape[0] - scaled_ais_layer.shape[0]), (0, 0)), mode="constant")

            if scaled_ais_layer.shape[1] < label.shape[1]:
                scaled_ais_layer = np.pad(scaled_ais_layer, ((0, 0), (0, label.shape[1] - scaled_ais_layer.shape[1]),), mode="constant")

            label[scaled_ais_layer == 1] = self.LABEL_SOURCE["ais"]

        # Process label source data
        if "vessel" in self.class_names:
            label[label == self.LABEL_SOURCE["ais"]] = self.LABELS["vessel"]

            if self.min_vessel_land_dist > 0:
                #label = cc.remove_vessels_close_to_land(label, distance_threshold=self.min_vessel_land_dist / self.downsampling_factor)
                label = cc.remove_vessels_close_to_land(label, distance_threshold=self.min_vessel_land_dist)
        else:
            label[label == self.LABEL_SOURCE["ais"]] = self.LABELS["background"]

        if self.unlabel_chart_data:
            label[label == self.LABEL_SOURCE["chart"]] = self.LABELS["unlabeled"]
        else:
            chart_data = (label == self.LABEL_SOURCE["chart"])
            #chart_classified = cc.classify_chart(chart_data,
            #                                     classes=[self.LABELS["islet"], self.LABELS["land"]],
            #                                     area_threshold=self.chart_area_threshold / self.downsampling_factor)
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
                    data = self.load_image(data_path)

                    if self.downsampling_factor > 1:
                        data = cv2.resize(data, None, fx=1 / self.downsampling_factor, fy=1 / self.downsampling_factor, interpolation=cv2.INTER_AREA)[:label.shape[0],:label.shape[1]]
                    else:
                        data = data[:label.shape[0], :label.shape[1]]
                else:
                    data = data[:label.shape[0], :label.shape[1]]

                chart_classified[(chart_data == 1) & (data < self.land_threshold)] = 0

                # TODO: possible speed up here with np.where or similar?
                if "islet" in self.class_names and "land" in self.class_names:
                    label = np.where((chart_classified != 0), chart_classified, label)
                elif "islet" in self.class_names:
                    label[chart_classified == self.LABELS["islet"]] = self.LABELS["islet"]
                else:
                    label[chart_classified != 0] = self.LABELS["land"]

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

    def update_cached_labels(self, components, collect_new=True):
        processed_labels = set()
        for entry in tqdm.tqdm(self.files[self.split], total=len(self.files[self.split]), desc="Updating cached labels", leave=False):
            label_path = self.get_label_path(entry["data"])
            if label_path in processed_labels:
                continue

            data_path = entry["data"]
            try:
                label = np.load(label_path).astype(np.int16)
            except (AttributeError, IOError) as e:
                print("Label does not exists at {}".format(label_path))
                if collect_new:
                    label = self.get_label(data_path)
                continue

            label = label[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]

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
                    processed_labels.add(label_path)
                    continue
                else:
                    ais = ais[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]
                label[ais == 1] = self.LABEL_SOURCE["ais"]
            if "chart" in components:
                land = self.data_loader.load_chart_layer_sensor(t, sensor, sensor_index, binary=True, only_first_range_step=True if self.width_region[1] <= 2000 else False)
                if isinstance(land, list):
                    self.logger.warning("Label {} not updated.\nChart data could not be gathered".format(label_path))
                    processed_labels.add(label_path)
                    continue
                else:
                    land = land[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]

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

    def get_distribution_statistics(self):
        missing_files = set()
        processed_files = set()
        mean_sum = np.zeros((2000), dtype=np.float64)
        std_sum = np.zeros((2000), dtype=np.float64)

        for i, entry in tqdm.tqdm(enumerate(self.files["train"]), total=len(self.files["train"])):
            if entry["data"] in processed_files:
                continue

            try:
                img = self.load_image(entry["data"])
                lbl = np.load(entry["label"])
            except DataFileNotFound:
                missing_files.add(entry)
                continue

            img = img[:, :2000].astype(np.float64)
            lbl = lbl[:img.shape[0], :img.shape[1]]
            img[lbl != 0] = np.nan
            mean_sum += np.nanmean(img, axis=0, dtype=np.float64)
            std_sum += np.nanmean(img, axis=0, dtype=np.float64)

            processed_files.add(entry["data"])

            if i % 1000 == 0:
                print("{}: mean: {}\tstd: {}".format(i, ",".join([str(v /len(processed_files)) for v in mean_sum]),
                                                     ",".join([str(v / len(processed_files)) for v in std_sum])))
        
        mean_sum /= len(processed_files)
        std_sum /= len(processed_files)
        np.save("mean_bgr.npy", mean_sum)
        np.save("std_bgr.npy", std_sum)
        return mean_sum, std_sum




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
            print("All input arguments must be positive integers")
            raise ValueError

        try:
            height_division_count = int(height_division_count)
            width_division_count = int(width_division_count)
            overlap = int(overlap)
        except:
            print("All input arguments must be positive integers")
            raise ValueError

        self.data_ranges = []
        h_step_size = int(round((self.height_region[1] - self.height_region[0]) / self.downsampling_factor) / (height_division_count + 1))
        w_step_size = int(round((self.width_region[1] - self.width_region[0]) / self.downsampling_factor) / (width_division_count + 1))

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

    def show_image_with_label(self, index=None, data_path=None, lbl=None, save_name=None, n_labels=4):
        if index is None and data_path is None:
            return

        if index is not None:
            data_path = self.files[self.split][index]["data"]
            data_range = self.data_ranges[self.files[self.split][index]["range"]]
            label_path = self.files[self.split][index]["label"]
        else:
            data_path = self.get_data_path(data_path)
            label_path = self.get_label_path(data_path)
            data_range = np.s_[self.height_region[0]:self.height_region[1], self.width_region[0]:self.width_region[1]]

        img = self.load_image(data_path)[data_range]
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, 2)

        if lbl is None:
            lbl = self.get_label(data_path, label_path, index=index)[data_range]

        #print(np.transpose(np.where(lbl == 1)))
        gt_count, gt_cc, gt_stats, gt_centroids = cc.get_connected_components(lbl == 1, connectivity=8)
        gt_centroids[:, 0], gt_centroids[:, 1] = gt_centroids[:, 1], gt_centroids[:, 0].copy()
        print(gt_centroids[1:, :])
        res = label2rgb(lbl, img, n_labels=n_labels)
        plt.imshow(res)
        plt.show()
        if save_name is not None:
            plt.imsave("figures/{}".format(save_name), res)
            res = cv2.resize(res, None, fx=1 / 9.9, fy=1 / 9.9 , interpolation=cv2.INTER_AREA)
            plt.imsave("figures/resize/{}".format(save_name), res,
                       format="png")


    def _colorbar(self, mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    def heatmap(self, data_path):
        img = self.load_image(data_path)[:, :2000]
        img = cv2.resize(img, None, fx=1 / 3.3, fy=1 / 3.3, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hmaps = np.load(osp.join("/data/polarlys/confidences", data_path.split("/")[-1].replace(".bmp", ".npy")))
        for hmap_i in range(hmaps.shape[0]):
            hmap = hmaps[hmap_i, :, :]
            if hmap.dtype != np.uint8:
                hmap = (hmap * 255).astype(np.uint8)
            hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
            res = cv2.addWeighted(hmap, 0.7, img, 0.3, 0)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            plt.imsave("figures/{}-{}".format(hmap_i, data_path.split("/")[-1].replace(".bmp", ".npy")), res, format="png")
            plt.imshow(res)
            plt.show()
            res = cv2.resize(res, None, fx= 1/3, fy=1/3, interpolation=cv2.INTER_AREA)
            plt.imsave("figures/resize/{}-{}".format(hmap_i, data_path.split("/")[-1].replace(".bmp", ".npy")), res, format="png")


if __name__ == "__main__":
    from polarlys.dataloader import DataLoader
    #np.s_[:int(4096/3), 0:2000], np.s_[int(4096/3):int(2*4096/3), 0:2000], np.s_[int(2*4096/3):, 0:2000]
    dataset = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets", cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt", split="train", dataset_name="test", min_data_interval=0, remove_files_without_targets=True, range_normalize=False)

    if True:
        #img, lbl = dataset[3]
        dataset.show_image_with_label(0, save_name=dataset.get_filename(0))
        #dataset.show_image_with_label(6, save_name=dataset.get_filename(6))
        exit(0)
    if True:
        dataset.heatmap(dataset.files["train"][6]["data"])
        exit(0)
    if False:
        #dataset.shuffle(0)
        with open("streak_verifier.txt", "a+") as index:
            for i in range(0, len(dataset)):
                dataset.show_image_with_label(index=i)
                res = input("Write to {} ({}) index? [y/n]".format(i, dataset.get_filename(i, True)))
                if res == "y":
                    index.write("{};\n".format(dataset.get_data_path(dataset.get_filename(i, True, True))))
                    index.flush()
    if True:
        t = datetime.datetime.strptime(dataset.get_filename(0), "%Y-%m-%d-%H_%M_%S")
        img = dataset.data_loader.load_image(t, 1, 0)
        img2 = dataset.data_loader.load_image(t + datetime.timedelta(seconds=10), 1, 0)
        img3 = dataset.data_loader.load_image(t + datetime.timedelta(seconds=300), 1, 0)
        print(dataset.get_filename(index=-1))
        exit(0)
    if False:
        dataset.radar_types = ["Radar1"]
        dataset.min_data_interval = 0
        dataset.find_streak_candidates(from_time=datetime.datetime(year=2018, month=1, day=3, hour=7))
    if False:
        dataset_streak = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets", cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt", split="test", dataset_name="streak5", min_data_interval=0, remove_files_without_targets=False, range_normalize=False)
        dataset.show_image_with_label(-1)
        exit(0)
    if False:
        exit(0)
        data_path = "/nas0/2017-11-24/2017-11-24-21/Radar0/2017-11-24-21_06_52.bmp"
        img = cv2.imread("/nas0/2017-11-24/2017-11-24-21/Radar0/2017-11-24-21_06_52.bmp", 0)
        #lbl = dataset.get_label(data_path)
        #print(np.transpose(np.where(lbl == 1)))
        #exit(0)
        #lbl = np.pad(lbl, ((0, 0), (0, 1400)), mode="constant")
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, 2)

        #res = label2rgb(lbl, img, n_labels=len(dataset.class_names))
        #plt.imshow(res)
        #plt.show()
        res = img

        basename = osp.splitext(data_path)[0]
        t = dataset.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = dataset.data_loader.get_sensor_from_basename(basename)
        y = dataset.data_loader.transform_image_from_sensor(t, sensor, sensor_index, dim=2828, scale_in=1, scale_out=3.39,
                                                                   image=res, use_gpu=False)
        #y = cv2.linearPolar(x, (0, 0), 2828, cv2.WARP_INVERSE_MAP)
        plt.imshow(y)
        plt.show()
        plt.imsave("radar_example_cart.png", y)
        exit(0)
    if False:
        print(dataset.get_mean())
        exit(0)
    if False:
        
        from scipy.optimize import curve_fit

        def func(x, a, b, c, d):
            return a*np.exp(-b*x) + c*np.exp(-d*x)

        mean_bgr = np.load("mean_bgr.npy")
        mean_bgr = mean_bgr[~np.isnan(mean_bgr)]
        xs = np.arange(start=0, stop=mean_bgr.shape[0], step=1)
        popt, pcov = curve_fit(func, xs, mean_bgr)

        long_x = np.arange(start=1438, stop=2000, step=1)
        new = func(long_x, *popt)
        new_mean_bgr = np.concatenate((mean_bgr, new))
        np.save("mean_bgr_filled.npy", new_mean_bgr)
        plt.plot(range(2000), new_mean_bgr)
        exit(0)
    if False:
        #print(dataset.get_distribution_statistics())
        #exit(0)
        img = dataset.load_image(dataset.files["train"][0]["data"])[:, :2000]
        plt.imsave("nonorm.png", img)

        #dataset.show_label(label=img)
        img = img.astype(np.float64)
        norm_img = dataset.apply_range_normalization(img)
        plt.imsave("norm1.png", norm_img)
        #dataset.show_label(label=norm_img)
        exit(0)
        ranges = [[0, 100], [100, 500], [500, 1000], [1000, 2000]]
        for i, r in enumerate(ranges):
            hist_homo, bins = np.histogram(norm_img[:, r[0]:r[1]].ravel(), 256, [-50, 200])
            plt.bar(bins[:-1], hist_homo)
            plt.title("Homo {}".format(" ".join(map(str, ranges[i]))))
            plt.show()
    
            hist, bins = np.histogram(img[:, r[0]:r[1]].ravel(), 256, [0, 256])
            plt.bar(bins[:-1], hist)
            plt.title("img {}".format(" ".join(map(str, ranges[i]))))
            plt.show()
        exit(0)
    if False:
        if True:
            files = dataset.files[dataset.split]
            new_files = []

            from_time = datetime.datetime(year=2017, month=10, day=27, hour=2, minute=0, second=0)
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
    if False:
        from skimage import exposure
        ranges = [[0, 5], [0, 10], [0, 15], [0, 20], [0, 25]]
        hist =[np.zeros((256), dtype=np.int64) for i in range(len(ranges))]
        for entry in tqdm.tqdm(dataset.files["train"], total=len(dataset.files["train"])):
            img = dataset.load_image(entry["data"])
            for i, r in enumerate(ranges):
                cumsum, bin_centers = exposure.cumulative_distribution(img[:, r[0]:r[1]])
                hist_img, bins = np.histogram(img[:, r[0]:r[1]].ravel(), 256, [0, 256])
                hist[i] += hist_img

        for i in range(len(hist)):
            plt.bar(bins[:-1], hist[i])
            plt.title(str(" ".join(map(str, ranges[i]))))
            plt.show()
            print(np.argmin(hist[i][hist[i] > 0]))
        exit(0)
    if False:
        img, lbl = dataset[0]
        exit(0)
        print("yes, this is doge")
        mean = 0
        pics = 0
        last_img = None
        for entry in tqdm.tqdm(dataset.files["train"], total=len(dataset.files["train"])):
            if entry["data"] != last_img:
                img = dataset.load_image(entry["data"])[:, 20:]
                last_img = entry["data"]

            homo_img = dataset.homomorphic_filtering(img=img[dataset.data_ranges[entry["range"]]].copy())
            mean += np.mean(homo_img)
            pics += 1

            if pics % 20 == 0:
                print("Mean so far: {}".format(mean/pics))
                #Mean so far: 55.59096748046876 after 20

        print("Mean: {}".format(mean/pics))
        exit(0)

        if False:
            print("hello?")
            ranges = [[0, 100], [0, 500], [0, 2000]]
            homo_img = dataset.homomorphic_filtering(dataset.files["train"][0]["data"])

            img = dataset.load_image(dataset.files["train"][0]["data"])[:1000, 25:2000]

            for i, r in enumerate(ranges):
                hist_homo, bins = np.histogram(homo_img[:, r[0]:r[1]].ravel(), 256, [0, 256])
                plt.bar(bins[:-1], hist_homo)
                plt.title("Homo {}".format(" ".join(map(str, ranges[i]))))
                plt.show()

                hist, bins = np.histogram(img[:, r[0]:r[1]].ravel(), 256, [0, 256])
                plt.bar(bins[:-1], hist)
                plt.title("img {}".format(" ".join(map(str, ranges[i]))))
                plt.show()
            plt.imshow(homo_img)
            plt.title("homo")
            plt.show()
            plt.imshow(img)
            plt.show()
            #img, lbl = dataset[0]
            #img = img[:, 25:]
            #dataset.show_label(label=img)
            exit(0)
        dataset = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                     cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt", split="valid",
                                     dataset_name="2018", remove_files_without_targets=True, height_divisions=0,
                                     width_divisions=0, min_data_interval=5*60)
        for idx in tqdm.tqdm(range(0, len(dataset)), total=len(dataset)):
            img, lbl = dataset[idx]
        exit()
    if False:
        dataset_collect = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                             cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg_lacie.txt",
                                             split="train", dataset_name="lacie_interval", min_data_interval=5*60,
                                             coordinate_system="Polar", height_divisions=0, width_divisions=0)
        dataset_collect.update_dataset_file(from_time=datetime.datetime(year=2018, month=1, day=9, hour=2, minute=0))
    elif False: # streak collecting
        dataset_collect = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                             cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg_lacie.txt",
                                             split="test", dataset_name="streak5", min_data_interval=0, height_divisions=0,
                                             width_divisions=0, radar_types=["Radar1"], remove_files_without_targets=True,
                                             skip_processed_files=False)
        dataset_collect.update_dataset_file(from_time=datetime.datetime(year=2018, month=1, day=31, hour=2, minute=47),
                                            to_time=datetime.datetime(year=2018, month=1, day=31, hour=2, minute=49))
    elif False:
        print(dataset.get_mean())
    elif False:
        files_to_update = []
        processed_files = set()
        dataset_valid = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                           cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt",
                                           split="valid", dataset_name="2018")
        all_files = dataset.files["train"]
        all_files.extend(dataset_valid.files["valid"])
        for file in tqdm.tqdm(all_files, total=len(all_files)):
            if file["data"] in processed_files:
                continue
            try:
                lbl = np.load(file["label"])
            except:
                files_to_update.append(file)
                processed_files.add(file["data"])
                print("Could not load label for {}".format(file["data"]))
                continue
            if not np.any(lbl == dataset.LABEL_SOURCE["chart"]):
                files_to_update.append(file)
            processed_files.add(file["data"])
        dataset.files["train"] = files_to_update
        with open("labels_to_update.txt", "a+") as cached_res:
            data_paths = [f["data"] for f in files_to_update]
            cached_res.writelines(data_paths)
        dataset.update_cached_labels(("ais", "chart", "unlabeled"))
    elif False:
        test = dataset.get_weather_data(dataset.files["train"][0]["data"])
        dataset.show_image_with_label(data_path="Radar0/2017-10-26-16_12_55.bmp")
        test = dataset.get_label("/nas0/2018-02-23/2018-02-23-13/Radar1/2018-02-23-13_00_04.bmp")
        for i in range(len(dataset.files["train"])):
            img, lbl = dataset[i]
        img, lbl = dataset[0]
    elif True:
        with open("../low_scoring_files_acc_cls_vessel.txt", "r") as file_index:
            lines = file_index.readlines()
            for line in lines:
                line = line.strip()
                print(line)
                dataset.show_image_with_label(data_path=line)
    elif False:
        processed_files = set()
        files_with_all_land = []
        dataset_valid = RadarDatasetFolder(root="/home/eivind/Documents/polarlys_datasets",
                                           cfg="/home/eivind/Documents/polarlys_datasets/polarlys_cfg.txt",
                                           split="valid", dataset_name="2018", HeightDivisons=0)
        all_files = dataset.files["train"]
        all_files.extend(dataset_valid.files["valid"])
        for file in tqdm.tqdm(all_files, total=len(all_files)):
            if file["data"] in processed_files:
                continue
            try:
                lbl = np.load(file["label"])
            except:
                processed_files.add(file["data"])
                print("Could not load label for {}".format(file["data"]))
                continue
            if np.all(lbl[:, 0] == dataset.LABEL_SOURCE["chart"]):
                files_with_all_land.append(file)
            processed_files.add(file["data"])
        with open("files_with_all_col0_land.txt", "a+") as cached_res:
            data_paths = [f["data"] for f in files_with_all_land]
            cached_res.writelines(data_paths)




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


