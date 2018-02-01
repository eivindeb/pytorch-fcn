import numpy as np
from math import exp
import timeit
import random
from os import path, listdir, makedirs
import shutil

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

def show_radar_data_with_label(data_path, label_path):
    here = path.dirname(path.abspath(__file__))
    from dataloader import DataLoader
    self_data_loader = DataLoader("/nas0", sensor_config=path.join(here, "dataloader.json"))
    # load image
    basename = path.splitext(data_path)[0]
    t = self_data_loader.get_time_from_basename(basename)
    sensor, sensor_index, subsensor_index = self_data_loader.get_sensor_from_basename(basename)

    img = self_data_loader.load_image(t, sensor, sensor_index, subsensor_index)
    label = np.load(label_path)

    # unlabel data blocked by mast for Radar0
    if sensor_index == 0:
        label[2000:2080, :] = -1

    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    res[:, :, 0] = img
    res[:, :, 1] = label == -1
    res[:,:,1] = res[:,:,1]*255

    res_img = Image.fromarray(res, 'RGB')
    res_img.show()

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

show_radar_data_with_label('/nas0/2017-10-28/2017-10-28-04/Radar1/2017-10-28-04_25_20.bmp','/data/polarlys/labels/2017-10-28/2017-10-28-04/Radar1/2017-10-28-04_25_20_label.npy')
#print(timeit.timeit("get_matrices()", number=100, globals=globals()))
#path1 = "/media/stx/2017/Radar0/test.jpg"
#path2 = "/mnt/stx/labels"

#get_file_and_label("/nas0/", "2017-10-24-15_10_01.bmp")
#remove_empty_files("/data/polarlys/")
#find_results_to_index_format()
#construct_directory_structure_and_move("/data/polarlys/labels/Radar1", "/data/polarlys/labels")
#print(timeit.timeit("load_vs_calc()", number=100, globals=globals()))
#npany()