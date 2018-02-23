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

# 10k MODEL-fcn32s_CFG-001_MOMENTUM-0.99_WEIGHT_DECAY-0.0005_INTERVAL_VALIDATE-6918_LR-7.000000000000001e-12_MAX_ITERATION-400000_VCS-b'5670d3d'_TIME-20171215-201543
# 5k ft MODEL-fcn32s_CFG-001_INTERVAL_VALIDATE-6918_WEIGHT_DECAY-0.0005_MOMENTUM-0.99_LR-7.000000000000001e-12_MAX_ITERATION-400000_VCS-b'5670d3d'_TIME-20171218-152416

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = "/data/polarlys"
    res_root = osp.join(root, "prediction_confidences_10k")
    data_folder = "/nas0/"


    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.RadarShipTargetFilterLandAndHidden(
            root, data_folder=data_folder, label_folder=osp.join(root, "labels"), split='test', transform=True,
            cache_labels=True, dataset_name="final", data_ranges=(np.s_[:, 0:2000],), min_data_interval=0, remove_files_without_targets=False),
        batch_size=1, shuffle=False, **kwargs)

    #test_loader.dataset.files["test"] = test_loader.dataset.files["test"][0:10]

    processed_files = []

    for i in range(len(test_loader.dataset.files["test"])):
        file_rel_path = test_loader.dataset.files["test"][i]["data"][0].replace(data_folder, "")
        save_location = osp.join(res_root, file_rel_path).replace(".bmp", "_confidence.npy")

        if osp.exists(save_location):
            processed_files.append(i)

    test_loader.dataset.files["test"] = [file for i, file in enumerate(test_loader.dataset.files["test"]) if i not in processed_files]

    n_class = len(test_loader.dataset.class_names)

    if osp.basename(model_file).startswith('fcn32s') or True:
        model = torchfcn.models.FCN32s(n_class=n_class)
    elif osp.basename(model_file).startswith('fcn16s'):
        model = torchfcn.models.FCN16s(n_class=n_class)
    elif osp.basename(model_file).startswith('fcn8s'):
        if osp.basename(model_file).startswith('fcn8s-atonce'):
            model = torchfcn.models.FCN8sAtOnce(n_class=n_class)
        else:
            model = torchfcn.models.FCN8s(n_class=n_class)
    else:
        raise ValueError
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
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader),
                                               total=len(test_loader),
                                               ncols=80, leave=False):

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
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

        confidence_ship = torch.nn.functional.softmax(score.data[0]).data[1].cpu().numpy()

        file_rel_path = test_loader.dataset.files["test"][batch_idx]["data"][0].replace(data_folder, "")
        save_location = osp.join(res_root, file_rel_path).replace(".bmp", "_confidence")
        if not osp.exists(osp.dirname(save_location)):
            os.makedirs(osp.dirname(save_location))

        np.save(save_location, confidence_ship)

        del score  # free up memory
        del target
        del data

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
