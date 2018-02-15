import numpy as np
from math import exp
import timeit
import random
from os import path, listdir, makedirs, statvfs
import shutil
import subprocess
import datetime

from PIL import Image

def linear_noise_gradient(r, c):
    max_val = 190
    min_val = -60
    for i in range(c.shape[1]):
        c[:, i] = int(-((max_val - min_val)/(c.shape[1] - 1))*i + max_val)
    return c

def exponential_noise_gradient(r, c):
    max_val = 190
    min_val = -60
    interval_length = max_val - min_val
    decay_constant = -10/c.shape[1]
    for i in range(c.shape[1]):
        c[:, i] = int(interval_length*exp(decay_constant*i) + min_val)

    return c


def test_disk_usage():
    disk_usage = statvfs("/data/polarlys/labels")
    size_of_filesystem = (disk_usage.f_frsize * disk_usage.f_blocks)/1024**3
    free_bytes = (disk_usage.f_frsize * disk_usage.f_bfree)/1024**3
    free_bytes_allowed = (disk_usage.f_frsize * disk_usage.f_bavail)/1024**3
    label_folder_size = subprocess.check_output(["du", "-sx", "/data/polarlys/labels"]).split()[0].decode("utf-8")
    print("hei")

def targets_in_range(data_path):
    from polarlys.dataloader import DataLoader
    self_data_loader = DataLoader("/nas0", sensor_config="/home/eivind/Documents/dev/sensorfusion/polarlys/dataloader.json")

    # load image
    basename = path.splitext(data_path)[0]
    t = self_data_loader.get_time_from_basename(basename)
    sensor, sensor_index = self_data_loader.get_sensor_from_basename(basename)

    ais_targets = self_data_loader.load_ais_targets_sensor(t, sensor, sensor_index)
    target_locations = [np.round(target[0], decimals=0).astype(np.int64).tolist() for target in ais_targets]

    if len(target_locations) > 0 and False:
        target_locations = [target[0] for target in ais_targets]
        target_locations_string = "/".join([str(np.round(target, decimals=0).astype(np.int64).tolist()) for target in target_locations])
        target_locations_read = []
        for target in target_locations_string.split("/"):
            target_split = target.strip("[]").split(",")
            target_locations_read.append([float(target_split[0]), float(target_split[1])])

        print("/".join([str(target) for target in target_locations_read]))
    else:
        print("[]")

    ais_targets = target_locations
    data_range = np.s_[:2200, :3000]
    for target in ais_targets:
        print(point_in_range(target[0], data_range, margin=200))
        print(target[0])

    print("********** ANY TARGETS IN RANGE: *******")
    print(any([point_in_range(target[0], data_range, margin=200) for target in ais_targets]))

    print("hei")

def point_in_range(point, range, margin=0):
    return ((range[0].start is None or point[0] - margin >= range[0].start) and (range[0].stop is None or point[0] + margin <= range[0].stop)) \
            and ((range[1].start is None or point[1] - margin >= range[1].start) and (range[1].stop is None or point[1] + margin <= range[1].stop))


def show_radar_data_with_label(data_path, label_path, cartesian=False):
    from polarlys.dataloader import DataLoader
    self_data_loader = DataLoader("/nas0", sensor_config="/home/eivind/Documents/dev/sensorfusion/polarlys/dataloader.json")
    # load image
    basename = path.splitext(data_path)[0]
    t = self_data_loader.get_time_from_basename(basename)
    sensor, sensor_index = self_data_loader.get_sensor_from_basename(basename)

    img = self_data_loader.load_image(t, sensor, sensor_index)
    label = np.zeros(img.shape)
    label[:, 0:2000] = np.load(label_path)

    # unlabel data blocked by mast for Radar0
    if sensor_index == 0:
        label[2000:2080, :] = -1

    img = img[:, 0:2000]
    label = label[:, 0:2000]

    threshold = 70
    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    res[:, :, 0] = img
    res[:, :, 1] = ((label == 2) & (img >= threshold))*100
    res[:,:,2] = (((label == 2) & (img < threshold)) | (label == 3))*50

    res_img = Image.fromarray(res, 'RGB')
    res_img = res_img.resize((int(res_img.size[0] * 0.25), int(res_img.size[1] * 0.25)),
                                       Image.ANTIALIAS)
    res_img.show()

    if cartesian:
        # label = self_data_loader.load_chart_layer_sensor(t, sensor, sensor_index, binary=True, only_first_range_step=True)

        to_transfer = np.dstack((img, label)).astype(np.int16)

        img_cart = self_data_loader.transform_image_from_sensor(t, sensor, sensor_index, image=to_transfer,
                                                                scale=3.39, dim=2828, use_gpu=False)

        label_cart = img_cart[:, :, 1].astype(np.int8)
        img_cart = img_cart[:, :, 0].astype(np.uint8)

        res_cart = np.zeros((img_cart.shape[0], img_cart.shape[1], 3), dtype=np.uint8)
        res_cart[:, :, 0] = img_cart
        res_cart[:, :, 1] = (label_cart == 2 & img > 50) * 255
        res_cart[:, :, 2] = (label_cart == 2) * 255

        res_img_cart = Image.fromarray(res_cart, 'RGB')
        res_img_cart = res_img_cart.resize((int(res_img_cart.size[0]*0.3), int(res_img_cart.size[1]*0.3)), Image.ANTIALIAS)
        res_img_cart.show()


def chart_difference_with_time(data_path):
    from polarlys.dataloader import DataLoader
    self_data_loader = DataLoader("/nas0",
                                  sensor_config="/home/eivind/Documents/dev/sensorfusion/polarlys/dataloader.json")
    # load image
    basename = path.splitext(data_path)[0]
    t_og = self_data_loader.get_time_from_basename(basename)
    print("Original time: {}".format(t_og))
    sensor, sensor_index = self_data_loader.get_sensor_from_basename(basename)
    label = np.load(path.join("/data/polarlys/labels", "/".join((basename + "_label.npy").split("/")[2:])))
    label = label[:, 0:2000]

    img = self_data_loader.load_image(t_og, sensor, sensor_index)

    img = img[:, 0:2000]
    # res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # res[:, :, 0] = img
    # res[:, :, 1] = (label == 1) * 255
    # res[:, :, 2] = (label == 2) * 255

    # res_img = Image.fromarray(res, 'RGB')
    # res_img = res_img.resize((int(res_img.size[0] * 0.2), int(res_img.size[1] * 0.2)),
    #                        Image.ANTIALIAS)
    # res_img.show()

    last_t = None

    for i in range(-20, 20):
        t = t_og + datetime.timedelta(seconds=i)
        if (True or abs(t_og - t) > datetime.timedelta(minutes=1)) and (
                last_t is None or abs(t - last_t) > datetime.timedelta(seconds=0)):
            last_t = t
            for j in range(2):
                img = self_data_loader.load_image(t, 1, j)
                if not isinstance(img, list):
                    basename = self_data_loader.get_filename_sec(t, "Radar" + str(j), "bmp")
                    label_og = np.load(path.join("/data/polarlys/labels",
                                                 "/".join(basename.replace(".bmp", "_label.npy").split("/")[2:])))
                    print("Radar{}: {}, delta: {}".format(j, t, (t - t_og).total_seconds()))
                    img = img[:, 0:2000]
                    label_og = label_og[:, 0:2000]
                    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    res[:, :, 0] = img
                    res[:, :, 1] = (label == 2) * 255
                    res[:, :, 2] = (label_og == 2) * 255

                    res_img = Image.fromarray(res, 'RGB')
                    res_img = res_img.resize((int(res_img.size[0] * 0.2), int(res_img.size[1] * 0.2)),
                                             Image.ANTIALIAS)
                    res_img.show()
                    break


def directory_structure_from_filename(root):
    with open("/data/polarlys/labels/index.txt", "r") as index:
        with open("/data/polarlys/labels/dataset_index.txt", "w") as dataset:
            new_lines = []
            for line in index.readlines():
                line = line.strip()
                line_split = line.split("/")
                radar_type = line_split[0]
                filename = line_split[1]
                file_times = filename.split("-")
                file_day = "{}-{}-{}".format(file_times[0], file_times[1], file_times[2])
                file_day_and_hour = "{}-{}".format(file_day, file_times[3].split("_")[0])
                new_lines.append("{}/{}/{}/{}/{}\n".format(root, file_day, file_day_and_hour, radar_type, filename))

            random.shuffle(new_lines)
            dataset.writelines(new_lines)


def construct_directory_structure_and_move(folder, new_root):
    for file in listdir(folder):
        if path.isfile(path.join(folder, file)):
            line_split = path.join(folder, file).split("/")
            radar_type = line_split[-2]
            filename = line_split[-1]
            file_times = filename.split("-")
            file_day = "{}-{}-{}".format(file_times[0], file_times[1], file_times[2])
            file_day_and_hour = "{}-{}".format(file_day, file_times[3].split("_")[0])
            abs_path = "{}/{}/{}/{}/{}".format(new_root, file_day, file_day_and_hour, radar_type, filename)
            last_dir = "/".join(abs_path.split("/")[0:-1])

            if not path.exists(last_dir):
                makedirs(last_dir)

            shutil.move(path.join(folder, file), abs_path)


def make_pred_heatmap():
    lbl_pred = np.load("lbl_pred.npy")
    score = np.load("score.npy")


    img_lbl_pred = Image.fromarray(np.uint8(lbl_pred))
    img_score = Image.fromarray(np.expand_dims(score, axis=2))
    img_lbl_pred.save("lbl_pred_img.png")
    img_score.save("score_img.png")




def get_mean_of_columns():
    m = np.random.randint(0, 10, (3, 10))
    s = np.mean(m, axis=0)
    print(m)
    print(s)
    print("hei")


def npany():
    test = np.zeros((10, 10))
    test[2, 2] = 1

    if np.any(test == 2):
        print("JA")
    else:
        print("NEI")


def write_random_files_from_index(index, new_index, number, from_time = None, to_time=None):
    if from_time is None:
        from_time = datetime.datetime(year=2000, month=1, day=1)
    if to_time is None:
        to_time = datetime.datetime(year=2100, month=1, day=1)

    with open(index, "r") as index_file, open(new_index, 'w') as new_index_file:
        old_lines = index_file.readlines()
        lines = []
        for line in old_lines:
            line_time = datetime.datetime.strptime(line.split(";")[0].split("/")[-1].replace(".bmp", ""), "%Y-%m-%d-%H_%M_%S")
            if from_time <= line_time <= to_time:
                lines.append(line)
        inds = random.sample(range(len(lines)), number)
        new_lines = []
        for i in inds:
            new_lines.append(lines[i])
        new_index_file.writelines(new_lines)

def find_results_to_index_format():
    with open("/data/polarlys/labels2/index_new.txt", "r") as find_results:
        with open("/data/polarlys/labels2/index.txt", "r") as old_results:
            old_lines = old_results.readlines()
        lines = [line for line in find_results.readlines() if line not in old_lines]
        new_lines = []
        for line in lines:
            line = line.strip()
            line_split = line.split("/")
            if len(line_split) != 5:
                continue

            new_lines.append(line[2:].replace("_label.npy", ".bmp;/\n"))

        random.shuffle(new_lines)

        with open("/data/polarlys/labels2/dataset_index.txt", "w+") as index:
            index.writelines(new_lines)


def load_vs_calc():
    #land = np.load("/data/polarlys/labels/Radar1/2017-10-22-11_25_53_label_land.npy")
    hidden_by_land = np.load("/data/polarlys/labels/Radar1/2017-10-22-11_25_53_label_land_hidden.npy")

    #hidden_by_land_mask = np.empty(land.shape, dtype=np.uint8)
    #hidden_by_land_mask[:, 0] = land[:, 0]
    #for col in range(1, land.shape[1]):
    #    np.bitwise_or(land[:, col], hidden_by_land_mask[:, col - 1], out=hidden_by_land_mask[:, col])


def get_matrices():
    pass
    #img = np.random.randint(0, 250, (4000, 4000))
    #channel = np.empty((1365, 2000, 3))
    #channel[:, :, 0] = img[0:1365, 0:2000]
    #temp1 = np.empty((1365, 2000))
    #temp2 = np.empty((1365, 2000))
    #max_val = 190
    #min_val = -60
    #interval_length = max_val - min_val
    #decay_constant = -10 / channel.shape[1]
    #for i in range(channel.shape[1]):
    #    temp1[:, i] = int(interval_length * exp(decay_constant * i) + min_val)
    #    temp2[:, i] = int(-((max_val - min_val) / (channel.shape[1] - 1)) * i + max_val)
    #channel[:, :, 1] = temp1
    #channel[:, :, 2] = temp2
    #lin = np.fromfunction(linear_noise_gradient, (1365, 2000))
    #expon = np.fromfunction(exponential_noise_gradient, (1365, 2000))
    #land = np.load("/data/polarlys/labels/Radar1/2017-10-22-11_25_53_label_land.npy")


def remove_target_information_from_index(index_path):
    with open(index_path, 'r+') as index:
        lines = index.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            line = line.split(";")[0]
            line = "{};\n".format(line)
            new_lines.append(line)

        index.seek(0)
        index.truncate()
        index.writelines(new_lines)


def get_file_and_label(data_folder, filename):
    here = path.dirname(path.abspath(__file__))
    from dataloader import data_loader
    self_data_loader = data_loader("/data/polarlys/", sensor_config=path.join(here, "dataloader.json"))
    filename_split = filename.split("-")
    file_rel_path = "{}/{}-{}/{}".format(filename_split[0], filename_split[1], filename_split[2], filename_split[3].split("_")[0])
    file_path = path.join(data_folder, file_rel_path)
    img = self_data_loader.load_image(file_path)
    ais = self_data_loader.load_ais_layer_by_basename(path.splitext(data_path)[0])



def remove_empty_files(root):
    here = path.dirname(path.abspath(__file__))
    from dataloader import data_loader
    self_data_loader = data_loader(root, sensor_config=path.join(here, "dataloader.json"))

    with open(path.join(root, "final_Radar1-Radar0_test.txt"), "r+") as index:
        lines = index.readlines()
        new_lines = []
        for line in lines:
            filename = line.split(";")[0]
            img = self_data_loader.load_image(path.join("/nas0/", filename))
            if not isinstance(img, list):
                new_lines.append(line)
        index.seek(0)
        index.truncate()
        index.writelines(new_lines)

file = "2017-11-24/2017-11-24-22/Radar0/2017-11-24-22_13_34"
file_label_path = path.join("/data/polarlys/labels/", file) + "_label.npy"
file_data_path = path.join("/nas0", file) + ".bmp"

#from_time = datetime.datetime(year=2017, month=11, day = 15)
#write_random_files_from_index("/home/eivind/Documents/polarlys_datasets/2018/train.txt", "/home/eivind/Documents/polarlys_datasets/sanity_check/train.txt", 2000, from_time)
#remove_target_information_from_index("/home/eivind/Documents/polarlys_datasets/2018/valid.txt")
#targets_in_range(file)
#test_disk_usage()
show_radar_data_with_label(file_data_path, file_label_path)
#print(timeit.timeit("get_matrices()", number=100, globals=globals()))
#path1 = "/media/stx/2017/Radar0/test.jpg"
#path2 = "/mnt/stx/labels"

#get_file_and_label("/nas0/", "2017-10-24-15_10_01.bmp")
#remove_empty_files("/data/polarlys/")
#find_results_to_index_format()
#construct_directory_structure_and_move("/data/polarlys/labels/Radar1", "/data/polarlys/labels")
#print(timeit.timeit("load_vs_calc()", number=100, globals=globals()))
#npany()