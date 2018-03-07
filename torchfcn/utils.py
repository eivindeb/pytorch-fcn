import numpy as np
import math
import skimage.segmentation
import scipy.spatial

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
      - mean accuracy
      - mean IU
      - fwavacc
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


def label_accuracy_score_from_hist(hist):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, fwavacc, mean_iu


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
                bjs.append("N/A")
            else:
                bjs.append(0)

    bjs.append(np.nanmean(bjs))
    return bjs
