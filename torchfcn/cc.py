import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage
from skimage import segmentation
from scipy import spatial
import math
import torch
import torchfcn

class InvalidClassesError(Exception):
    pass

def get_f1_score(data, classes, img=None):
    connectivity = 8
    for class_name, settings in classes.items():
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats((data == settings["image_value"]).astype(np.uint8), connectivity)
        print("tjohei")
    print("hei")


def plot(cc_labels, label=None, img=None):
    subplot_count = 1
    subplot_count += 1 if label is not None else 0
    subplot_count += 1 if img is not None else 0
    if subplot_count == 1:
        plt_im = plt.imshow(cc_labels)
        colorbar(plt_im)
    elif subplot_count == 2:
        f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw={"xticks": [], "yticks": []})
        ax0.imshow(img if img is not None else label)
        ax1_im = ax1.imshow(cc_labels)
        colorbar(ax1_im)
    elif subplot_count == 3:
        f, (ax0, ax1, ax2) = plt.subplots(1, 3, subplot_kw={"xticks": [], "yticks": []})
        ax0_im = ax0.imshow(img)
        #colorbar(ax0_im)
        ax1_im = ax1.imshow(label)
        #colorbar(ax1_im)
        ax2_im = ax2.imshow(cc_labels)
        #colorbar(ax2_im)

    plt.show()


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def classify_chart(data, classes, area_threshold):
    """
    :param data: Binary data where 1 represents chart data, and 0 background
    :param classes: The image value of the two classes to classify chart data in [under_threshold, over_threshold]
    :param area_threshold: Connected component area threshold used to classify chart into classes
    :return: Data where each 1 in input is classified into(label == 2).astype(np.uint8) one of the two classes
    """
    if not isinstance(classes, list) or len(classes) != 2:
        raise InvalidClassesError
    connectivity = 8

    if data.dtype == np.dtype(np.bool):
        data = data.astype(np.uint8)

    retval, cc_labels, stats, centroids = get_connected_components(data, connectivity)
    data = data * classes[0]
    data[(cc_labels != 0) & (stats[cc_labels][:, :, 4] >= area_threshold)] = classes[1]
    return data


def get_connected_components(data, connectivity):
    if data.dtype == np.dtype(np.bool):
        data = data.astype(np.uint8)
    return cv2.connectedComponentsWithStats(data, connectivity)  # index 0 corresponds to background


def boundary(data, data_pred, classes=None):
    if classes is None:
        classes = range(1, data.max() + 1 if data.max() >= data_pred.max() else data_pred.max() + 1)
    theta = math.sqrt(data.shape[0] ** 2 + data.shape[1] ** 2) * 0.0075

    bjs = []
    for class_val in classes:
        boundary_pred = segmentation.find_boundaries(data_pred == class_val, connectivity=1, mode="inner", background=0)
        boundary_pred_length = np.count_nonzero(boundary_pred)
        boundary_label = segmentation.find_boundaries(data == class_val, connectivity=1, mode="inner", background=0)
        boundary_label_length = np.count_nonzero(boundary_label)
        if boundary_pred_length > 0 and boundary_label_length > 0:
            boundary_label_coords = np.transpose(np.where(boundary_label))
            boundary_pred_coords = np.transpose(np.where(boundary_pred))
            non_overlapping_pred_coords = np.transpose(np.where((boundary_pred) & (data != class_val)))
            non_overlapping_label_coords = np.transpose(np.where((boundary_label) & (data_pred != class_val)))

            # First calculate true positives for prediction to label
            mytree = spatial.cKDTree(boundary_label_coords)
            TP_p = boundary_pred_length - non_overlapping_pred_coords.shape[0]
            for point in non_overlapping_pred_coords:
                dist, idx = mytree.query(point, distance_upper_bound=theta)
                if dist < theta:
                    TP_p += 1 - (dist/theta) ** 2
            FN = boundary_pred_length - TP_p

            # Then for label to prediction
            mytree = spatial.cKDTree(boundary_pred_coords)
            TP_gt = boundary_label_length - non_overlapping_label_coords.shape[0]
            for point in non_overlapping_label_coords:
                dist, idx = mytree.query(point, distance_upper_bound=theta)
                if dist < theta:
                    TP_gt += 1 - (dist/theta) ** 2
            FP = boundary_label_length - TP_gt

            TP = TP_gt + TP_p

            bjs.append(TP / (TP + FP + FN))
        else:
            bjs.append(0)

    return bjs


if __name__ == "__main__":
    rel_path = "2017-11-24/2017-11-24-19/Radar1/2017-11-24-19_57_41"
    label_folder = "/data/polarlys/labels/"
    data_folder = "/nas0"
    if False:
        with open("/home/eivind/Documents/polarlys_datasets/2018/train.txt", "r") as index:
            lines = index.readlines()[100:]
            land_areas = []
            #area_bins =
            for line_num, line in enumerate(lines):
                rel_path = line.split(";")[0]
                label = np.load(osp.join(label_folder, rel_path.replace(".bmp","_label.npy")))
                #data_img = cv2.imread(osp.join(data_folder, rel_path)[:, :2000]
                retval, cc_labels, stats, centroids = get_connected_components((label == 2).astype(np.uint8), 8)
                land_areas.extend([stat[4] for i, stat in enumerate(stats) if i != 0])
                label_10k = np.copy(label)
                label_5k = np.copy(label)
                label_test = np.copy(label)
                chart_classified = classify_chart((label == 2).astype(np.uint8), [2, 4], 10000)
                #label_test[chart_classified == 2] = 2
                #label_test[chart_classified == 4] = 4
                label_test = np.where(chart_classified != 0, chart_classified, label)

                label_10k[(cc_labels != 0) & (stats[cc_labels][:, :, 4] >= 10000)] = 4
                label_5k[(cc_labels != 0) & (stats[cc_labels][:, :, 4] >= 5000)] = 4
                if line_num % 19 == 0 or True:
                    plot(label_test, label=label_5k, img=label_10k)
                if line_num == 100:
                    break
            hist, bins = np.histogram(land_areas, bins=[2500, 5000, 10000, 15000, 20000, 25000, 40000, 100000])
            #plt.hist(np.asarray(land_areas), bins="auto")
            plt.plot(bins[:-1], hist)
            plt.show()
            print("hei")
    elif False:
        label = np.zeros((4096, 2000), dtype=np.uint8)
        label[1000:1500, 1000:1500] = 1
        label[3050: 3450, 1800:1900] = 1
        retval, cc_labels, stats, centroids = get_connected_components(label, 8)
        get_f1_score(label, {"land": {"image_value": 2}}, img=data_img)
    elif True:
        root = "/home/eivind/Documents/polarlys_datasets"
        dataset = torchfcn.datasets.RadarDatasetFolder(root, split='valid', cfg=osp.join(root, "polarlys_cfg.txt"), transform=False, dataset_name="2018")
        img, lbl = dataset[1]
        pred = np.load("/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_LR-3.5000000000000004e-12_INTERVAL_WEIGHT_UPDATE-10_MOMENTUM-0.99_INTERVAL_VALIDATE-30000_INTERVAL_CHECKPOINT-500_WEIGHT_DECAY-0.0005_MAX_ITERATION-800000_VCS-b'39e5f59'_TIME-20180302-152858/results/Radar1/2017-11-25-03_51_07.npy")
        label = np.load(osp.join(label_folder, "2017-10-28/2017-10-28-00/Radar1/2017-10-28-00_40_00" + "_label.npy"))
        pred = pred[0, :, :]
        data_ranges = [np.s_[:int(4096/3), :], np.s_[int(4096/3):int(4096*2/3), :], np.s_[int(4096*2/3):, :]]
        for d in data_ranges:
            print(boundary(lbl[d], pred[d]))

