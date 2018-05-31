#!/usr/bin/env python

import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchfcn
import tqdm
from fcn.utils import label2rgb
import matplotlib.pyplot as plt
import pss.models.psp_net as psp_net
import yaml

# 10k MODEL-fcn32s_CFG-001_MOMENTUM-0.99_WEIGHT_DECAY-0.0005_INTERVAL_VALIDATE-6918_LR-7.000000000000001e-12_MAX_ITERATION-400000_VCS-b'5670d3d'_TIME-20171215-201543
# 5k ft MODEL-fcn32s_CFG-001_INTERVAL_VALIDATE-6918_WEIGHT_DECAY-0.0005_MOMENTUM-0.99_LR-7.000000000000001e-12_MAX_ITERATION-400000_VCS-b'5670d3d'_TIME-20171218-152416

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #model_file = args.model_file

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = "/home/eivind/Documents/polarlys_datasets/"
    model_folder = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final6_MODEL-PSPnet_TIME-20180521-133432"
    model_file = osp.join(model_folder, "model_best.pth.tar")
    data_folder = "/nas0/"
    #results_folder = osp.join(model_folder, "results")
    results_folder = "/data/polarlys/confidences"

    with open(osp.join(model_folder, "config.yaml"), 'r') as f:
        cfg = yaml.load(f)


    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.RadarDatasetFolder(
            root, split='train', cfg=osp.join(model_folder, "polarlys_cfg_valid.txt"), transform=True, dataset_name="test", min_data_interval=0, remove_files_without_targets=False),
        batch_size=1, shuffle=False, **kwargs)

    n_class = len(test_loader.dataset.class_names)
    metadata = test_loader.dataset.metadata

    if osp.basename(model_file).startswith('fcn32s') or False:
        model = torchfcn.models.FCN32s(n_class=n_class)
    elif osp.basename(model_file).startswith('fcn16s'):
        model = torchfcn.models.FCN16s(n_class=n_class)
    elif osp.basename(model_file).startswith('fcn8s') or False:
        if osp.basename(model_file).startswith('fcn8s-atonce') or True:
            model = torchfcn.models.FCN8sAtOnce(n_class=n_class, metadata=metadata)
        else:
            model = torchfcn.models.FCN8s(n_class=n_class)
    else:
        model = psp_net.PSPNet(num_classes=n_class, pretrained=cfg["pretrained"],
                               metadata_channels=14 if test_loader.dataset.metadata else 0,
                               in_channels=1 if test_loader.dataset.image_mode == "Grayscale" else 3,
                               use_cfar_filters=cfg["cfar"], use_aux=True, freeze=None, group_norm=cfg["group_norm"])
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with Radar test set')
    visualizations = []
    label_trues, label_preds = [], []
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(test_loader),
                                                   total=len(test_loader),
                                                   ncols=80, leave=False):

            #if test_loader.get_filename(batch_idx, with_radar=True).split("/")[-1] == "Radar0":
            #    continue
            if metadata:
                if cuda:
                    data_img, data_meta, target = batch[0].cuda(), batch[2].cuda(), batch[1].cuda()
                else:
                    data_img, data_meta, target = batch[0], batch[2], batch[1]
                data_img, data_meta, target = Variable(data_img, volatile=True), Variable(data_meta,
                                                                                          volatile=True), Variable(target)
                score = model(data_img, data_meta)
            else:
                if cuda:
                    data, target = batch[0].cuda(), batch[1].cuda()
                else:
                    data, target = batch[0], batch[1]
                data, target = Variable(data), Variable(target)
                score = model(data)


            #imgs = data.data.cpu()
            #lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            #lbl_true = target.data.cpu()

            #for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            #    img, lt = test_loader.dataset.untransform(img, lt)
            #    label_trues.append(lt)
            #    label_preds.append(lp)
            #    if len(visualizations) < 9:
            #        viz = fcn.utils.visualize_segmentation(
            #            lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
            #            label_names=test_loader.dataset.class_names)
            #        visualizations.append(viz)

            #lbl_preds = score.data.max(1)[1].cpu().numpy()[:, :, :]
            confidence_ship = torch.nn.functional.softmax(score)[0, :, :, :].cpu().numpy()

            filename = test_loader.dataset.get_filename(batch_idx, with_radar=False)
            save_location = osp.join(results_folder, filename)
            if not osp.exists(osp.dirname(save_location)):
                os.makedirs(osp.dirname(save_location))

            np.save(save_location, confidence_ship)

            continue

            lt = target.data.cpu()
            imgs = data.data.cpu() if not metadata else data_img.data.cpu()
            imgs, lbl_trues = test_loader.dataset.untransform(imgs, lt)
            for img, lbl_true, lbl_pred in zip(imgs, lbl_trues, lbl_preds):
                if len(img.shape) == 2:
                    img = np.repeat(img[:, :, np.newaxis], 3, 2)

                res = label2rgb(lbl_true, img, n_labels=n_class)
                plt.imsave(osp.join(results_folder, test_loader.get_filename(batch_idx)), res)
                """
                f, (ax0, ax1) = plt.subplots(1, 2)
                ax0.imshow(label2rgb(lbl_true, img, n_labels=n_class))
                ax0.set_title("Ground truth")
                ax1.imshow(label2rgb(lbl_pred, img, n_labels=n_class))
                ax1.set_title("Prediction")
                plt.show()
                """

    #metrics = torchfcn.utils.label_accuracy_score(
    #    label_trues, label_preds, n_class=n_class)
    #metrics = np.array(metrics)
    #metrics *= 100
    #print('''\
#Accuracy: {0}
#Accuracy Class: {1}
#Mean IU: {2}
#FWAV Accuracy: {3}'''.format(*metrics))

    #viz = fcn.utils.get_tile_image(visualizations)
    #skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
