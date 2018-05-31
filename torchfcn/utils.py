import numpy as np
import math
import skimage.segmentation
import scipy.spatial
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm as cmap
import itertools

# confusion matrix !!
def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class, per_class=True):
    """Returns accuracy score evaluation result.

      - overall accuracy
      (- per class accuracy)
      - mean accuracy
      - fwavacc
      (- per class IU)
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls_mean = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if per_class:
        return acc, acc_cls, acc_cls_mean, fwavacc, iu, mean_iu
    else:
        return acc, acc_cls_mean, fwavacc, mean_iu


def label_accuracy_score_from_hist(hist, per_class=True):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls_mean = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if per_class:
        return acc, acc_cls, acc_cls_mean, fwavacc, iu, mean_iu
    else:
        return acc, acc_cls_mean, fwavacc, mean_iu


def boundary_jaccard(lbl_true, lbl_pred, classes=None):
    if len(lbl_true.shape) == 3:
        if lbl_true.shape[0] != 1:
            print("Calculation of boundary jaccard index with batch size larger than 1 not implemented")
            raise NotImplementedError
        else:
            lbl_true = lbl_true[0, :, :]
            lbl_pred = lbl_pred[0, :, :]

    if classes is None:
        classes = range(lbl_true.max() + 1 if lbl_true.max() >= lbl_pred.max() else lbl_pred.max() + 1)
    theta = math.sqrt(lbl_true.shape[0] ** 2 + lbl_true.shape[1] ** 2) * 0.0075

    bjs = []
    for class_val in classes:
        class_gt = lbl_true == class_val
        class_pred = lbl_pred == class_val
        if np.any(class_gt) and np.any(class_pred):
            boundary_pred = skimage.segmentation.find_boundaries(class_pred, connectivity=1, mode="inner", background=0)
            boundary_pred_length = np.count_nonzero(boundary_pred)

            if boundary_pred_length == 0:  # entire label is class, therefore no boundary
                boundary_pred = np.ones((boundary_pred.shape))
                boundary_pred[1:-1, 1:-1] = 0
                boundary_pred = boundary_pred.astype(np.bool)
                boundary_pred_length = np.count_nonzero(boundary_pred)

            boundary_label = skimage.segmentation.find_boundaries(class_gt, connectivity=1, mode="inner", background=0)
            boundary_label_length = np.count_nonzero(boundary_label)

            if boundary_label_length == 0:
                boundary_label = np.ones((boundary_label.shape))
                boundary_label[1:-1, 1:-1] = 0
                boundary_label = boundary_label.astype(np.bool)
                boundary_label_length = np.count_nonzero(boundary_label)

            boundary_label_coords = np.transpose(np.where(boundary_label))
            boundary_pred_coords = np.transpose(np.where(boundary_pred))

            non_overlapping_pred_coords = np.transpose(np.where((boundary_pred) & (lbl_true != class_val)))
            non_overlapping_label_coords = np.transpose(np.where((boundary_label) & (lbl_pred != class_val)))

            # First calculate true positives for prediction to label
            mytree = scipy.spatial.cKDTree(boundary_label_coords)
            TP_p = boundary_pred_length - non_overlapping_pred_coords.shape[0]
            dists, idxs = mytree.query(non_overlapping_pred_coords, distance_upper_bound=theta)
            dists = np.where(dists < theta, 1 - (dists/theta) ** 2, 0)
            TP_p += np.sum(dists)
            FN = boundary_pred_length - TP_p

            # Then for label to prediction
            mytree = scipy.spatial.cKDTree(boundary_pred_coords)
            TP_gt = boundary_label_length - non_overlapping_label_coords.shape[0]

            dists, idxs = mytree.query(non_overlapping_label_coords, distance_upper_bound=theta)
            dists = np.where(dists < theta, 1 - (dists / theta) ** 2, 0)
            TP_gt += np.sum(dists)

            FP = boundary_label_length - TP_gt

            TP = TP_gt + TP_p

            bjs.append(TP / (TP + FP + FN))
        else:
            if not np.any(class_gt) and not np.any(class_pred):
                bjs.append(np.nan)
            else:
                bjs.append(0)

    bjs.append(np.nanmean(bjs))
    return bjs

def visualize_confusion_matrix(cm_path, classes, normalize=True):
    cm = np.load(cm_path)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j ] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("cm.pdf", format="pdf")
    plt.show()

def plot_roc():
    import os.path as osp
    data_folder = "/data/polarlys/ts1"
    ml_pd = np.load(osp.join(data_folder, "ml/p_ds.npy"))
    ml_pfa = np.load(osp.join(data_folder, "ml/p_fas.npy"))
    cfar_pd = np.load(osp.join(data_folder, "cfar/p_ds.npy"))
    cfar_pfa = np.load(osp.join(data_folder, "cfar/p_fas.npy"))
    x_scale = "lin"

    plt.plot(ml_pfa, ml_pd)
    plt.plot(cfar_pfa, cfar_pd)
    if x_scale == "log":
        plt.xscale("log")
        plt.xlim([min([min(ml_pfa), min(cfar_pfa)]), 1])
    else:
        plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(["ML", "CFAR"])
    plt.title("ROC TS1 Linear")
    plt.xlabel("False alarm rate")
    plt.ylabel("Probability of detection")
    plt.savefig("roc_ts1_{}.pdf".format(x_scale), bbox_inches="tight", format="pdf")
    plt.show()

def plot_streak_histogram():
    ml = np.load("/data/polarlys/streaks/ml_0.001_streaks.npy")
    cfar = np.load("/data/polarlys/streaks/cfar_150_streaks.npy")
    highest = max(np.max(ml[:, 0]), np.max(cfar[:, 0]))
    cfar_hist = np.zeros(highest + 1)
    cfar_hist[cfar[:, 0]] = cfar[:, 1]

    ml_hist = np.zeros(highest + 1)
    ml_hist[ml[:, 0]] = ml[:, 1]

    plt.bar(range(1, highest + 2), cfar_hist, alpha=0.5)
    plt.bar(range(1, highest + 2), ml_hist, alpha=0.5)
    plt.legend(["CFAR", "ML"])
    plt.savefig("streaks_pfa_0.006.pdf", format="pdf")
    plt.show()


class CFARModule(nn.Module):
    def __init__(self, guard_band_size, reference_band_size, alpha):
        super(CFARModule, self).__init__()
        self.reference_band = reference_band_size
        self.guard_band = guard_band_size
        self.alpha = alpha
        self.pad = reference_band_size + guard_band_size
        self.padder = nn.ReplicationPad2d(self.pad)
        self.padder.requires_grad = False

        self.filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=2*(guard_band_size + reference_band_size) + 1,
                                    padding=0)
        self.initialize_weights()

    # TODO: scale output or batchnorm? check that loading of parameters works, add support for freezing

    def forward(self, x):
        x = self.padder(x)
        x = self.filter(x)
        x_min, x_max = x.min(), x.max()
        x = ((195 - (-59))*(x - x_min)) / (x_max - x_min) + (-59)
        return x


    def initialize_weights(self):
        filter_size = 2 * (self.guard_band + self.reference_band) + 1
        weights = torch.FloatTensor(1, 1, filter_size, filter_size).uniform_(-1.01, -0.99)
        weights[0, 0, self.reference_band:filter_size - self.reference_band,
        self.reference_band:filter_size - self.reference_band].uniform_(0, 0.001)
        weights[0, 0, filter_size // 2, filter_size // 2] = abs(weights.sum()) / self.alpha

        self.filter.weight = torch.nn.Parameter(weights)


class MetadataModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MetadataModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)


class GroupNorm(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupNorm,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta


def flatten(l):
    for el in l:
        try:
            yield from flatten(el)
        except TypeError:
            yield el


if __name__ == "__main__":
    plot_streak_histogram()
    exit(0)
    if True:
        cm_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final6_MODEL-PSPnet_TIME-20180521-133432/cm/iter480000.npy"
        classes = ["Background", "Vessel", "Land", "Unknown"]
        visualize_confusion_matrix(cm_path, classes)
        #visualize_confusion_matrix(cm_path, classes, normalize=False)
    if False:
        from collections import Iterable
        import os.path as osp
        rel_path = "2017-11-24/2017-11-24-19/Radar1/2017-11-24-19_57_41"
        label_folder = "/data/polarlys/labels/"
        data_folder = "/nas0"
        lbl = np.load(osp.join(label_folder, rel_path + "_label.npy"))
        #boundary_jaccard(lbl, lbl)
        test = np.asarray([0.5, 0.3, 5, np.nan, 0.3])
        print(test)
        test_m = np.nanmean(test)
        print(test_m)

