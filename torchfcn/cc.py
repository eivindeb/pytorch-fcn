import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        colorbar(ax0_im)
        ax1_im = ax1.imshow(label)
        colorbar(ax1_im)
        ax2_im = ax2.imshow(cc_labels)
        colorbar(ax2_im)

    plt.show()


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def classify_chart(data, area_threshold):
    connectivity = 8


def get_connected_components(data, connectivity):
    return cv2.connectedComponentsWithStats(data, connectivity)  # index 0 corresponds to background


if __name__ == "__main__":
    rel_path = "2017-11-24/2017-11-24-19/Radar1/2017-11-24-19_57_41"
    label_folder = "/data/polarlys/labels/"
    data_folder = "/nas0"
    if True:
        with open("/home/eivind/Documents/polarlys_datasets/2018/train.txt", "r") as index:
            lines = index.readlines()
            land_areas = []
            #area_bins =
            for line_num, line in enumerate(lines):
                rel_path = line.split(";")[0]
                label = np.load(osp.join(label_folder, rel_path.replace(".bmp","_label.npy")))
                #data_img = cv2.imread(osp.join(data_folder, rel_path)[:, :2000]
                retval, cc_labels, stats, centroids = get_connected_components((label == 2).astype(np.uint8), 8)
                land_areas.extend([stat[4] for i, stat in enumerate(stats) if i != 0])
                new_label = np.copy(label)
                new_label[(cc_labels != 0) & (stats[cc_labels][:, :, 4] < 10000)] = 4
                plot(cc_labels, label=label, img=new_label)
                if line_num == 5:
                    break
            hist, bins = np.histogram(land_areas)
            #plt.hist(np.asarray(land_areas), bins="auto")
            plt.plot(hist, bins[:-1])
            plt.show()
            print("hei")
    elif False:
        label = np.zeros((4096, 2000), dtype=np.uint8)
        label[1000:1500, 1000:1500] = 1
        label[3050: 3450, 1800:1900] = 1
        retval, cc_labels, stats, centroids = get_connected_components(label, 8)
        get_f1_score(label, {"land": {"image_value": 2}}, img=data_img)

