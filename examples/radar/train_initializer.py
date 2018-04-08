#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import pytz
import torch
import yaml

import time

import torchfcn
import shutil
import ptsemseg.models.pspnet as PSPNet
import ptsemseg.models.linknet as LinkNet
import ptsemseg.models.unet as Unet
import pss.models.psp_net as psp_net
import pss.models.gcn as gcn
from ml.pytorch_refinenet.pytorch_refinenet.refinenet import RefineNet4CascadePoolingImproved as RefineNet

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    "fcn": dict(
        max_iteration=800000,
        lr=5e-11*0.07,  # the standard learning rate for VOC images (500x375) multiplied by ratio of radar dataset image size (1365x2000)
        lr_decay=0.1,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=30000,
        interval_checkpoint=500,  # checkpoint every 10 minutes
        interval_weight_update=1,
    ),
    "PSPnet2": dict(
        max_iteration=800000,
        lr=5e-11*0.07,  # the standard learning rate for VOC images (500x375) multiplied by ratio of radar dataset image size (1365x2000)
        lr_decay=0.1,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=30000,
        interval_checkpoint=500,  # checkpoint every 10 minutes
        interval_weight_update=1,
    ),
    "PSPnet": dict(
        max_iteration=800000,
        lr=1e-10,  # the standard learning rate for VOC images (500x375) multiplied by ratio of radar dataset image size (1365x2000)
        lr_decay=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        interval_validate=30000,
        interval_checkpoint=500,  # checkpoint every 10 minutes
        interval_weight_update=16,
        pretrained=False,
    ),
    "GCN": dict(
        max_iteration=800000,
        lr=1e-17,  # the standard learning rate for VOC images (500x375) multiplied by ratio of radar dataset image size (1365x2000)
        lr_decay=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        interval_validate=30000,
        interval_checkpoint=500,  # checkpoint every 10 minutes
        interval_weight_update=16,
        pretrained=False,
    ),
    "RefineNet": dict(
        max_iteration=800000,
        lr=7e-11,  # the standard learning rate for VOC images (500x375) multiplied by ratio of radar dataset image size (1365x2000)
        lr_decay=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        interval_validate=30000,
        interval_checkpoint=500,  # checkpoint every 10 minutes
        interval_weight_update=1,
        pretrained=False,
    ),
}


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, cfg):
    # load config
    name = 'MODEL-%s' % (model_name)
    now = datetime.datetime.now(pytz.timezone('Europe/Oslo'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))


def main():
    model_name = "RefineNet"
    dataset_name = "test"

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    resume = args.resume

    if "fcn" in model_name:
        fcn = True
        model_cfg = "fcn"
    else:
        fcn = False
        model_cfg = model_name

    root = "/home/eivind/Documents/polarlys_datasets"

    if resume:
        checkpoint = torch.load(resume)
        out = checkpoint['out']
        dataset_name = checkpoint.get("dataset_name", "test")
        with open(osp.join(out, "config.yaml"), 'r') as f:
            cfg = yaml.load(f)
        dataset_cfg = osp.join(out, "polarlys_cfg.txt")
    else:
        dataset_cfg = osp.join(root, "polarlys_cfg.txt")
        cfg = configurations[model_cfg]#configurations[args.config]
        out = get_log_dir(model_name, cfg)

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    #root = "/media/stx/LaCie1/export"

    #root = osp.expanduser('~/data/datasets/Radar')

    data_folder = "/nas0/"
    #data_folder = root

    if model_name == "PSPnet":
        radar_kwargs = {"train": {"height_divisions": 7, "width_divisions": 0}, "valid": {"height_divisions": 1, "width_divisions": 0}}
    elif fcn:
        radar_kwargs = {"train": {"height_divisions": 2, "width_divisions": 0}, "valid": {"height_divisions": 1, "width_divisions": 0}}
    elif model_name == "GCN":
        radar_kwargs = {"train": {"height_divisions": 5, "width_divisions": 0, "overlap": 0}}
        radar_kwargs.update({"valid": radar_kwargs["train"]})
    elif model_name == "RefineNet":
        radar_kwargs = {"train": {"height_divisions": 5, "width_divisions": 0, "overlap": 0, "image_height": 4032, "image_width": 1984}}
        radar_kwargs.update({"valid": radar_kwargs["train"]})
    elif model_name == "LinkNet":
        radar_kwargs = {"train": {"height_divisions": 5, "width_divisions": 0, "overlap": 0}}
        radar_kwargs.update({"valid": radar_kwargs["train"]})
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.RadarDatasetFolder(
            root, split='train', cfg=dataset_cfg, transform=True, dataset_name=dataset_name, min_data_interval=10, **radar_kwargs["train"]),
        batch_size=1, shuffle=False, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.RadarDatasetFolder(
            root, split='valid', cfg=dataset_cfg, transform=True, dataset_name=dataset_name, min_data_interval=60*5, **radar_kwargs["valid"]),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    n_class = train_loader.dataset.class_names.size

    if model_name == "PSPnet":
        model = psp_net.PSPNet(num_classes=n_class, pretrained=cfg["pretrained"], metadata_channels=14 if train_loader.dataset.metadata else 0)
    elif model_name == "fcn32s":
        model = torchfcn.models.FCN32s(n_class=n_class, metadata=train_loader.dataset.metadata)
    elif model_name == "fcn8s":
        model = torchfcn.models.FCN8sAtOnce(n_class=n_class, metadata=train_loader.dataset.metadata)
    elif model_name == "GCN":
        model = gcn.GCN(num_classes=n_class, pretrained=cfg["pretrained"], input_size=(682, 2000))
    elif model_name == "LinkNet":
        model = LinkNet(n_classes=n_class, in_channels=1)
    elif model_name == "RefineNet":
        model = RefineNet((1, 1024, 1984), num_classes=n_class, pretrained=False, freeze_resnet=False)

    start_epoch = 0
    start_iteration = 0
    if resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        if fcn:
            vgg16 = torchfcn.models.VGG16(pretrained=False)
            model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    if fcn:
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        optim = torch.optim.SGD([
            {"params": [param for name, param in model.named_parameters() if name[-4:] == "bias"], "lr": 2 * cfg["lr"]},
            {"params": [param for name, param in model.named_parameters() if name[-4:] != "bias"], "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"]}
            ],  momentum=cfg['momentum'],
                nesterov=True
        )

    schedul = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=cfg["lr_decay"], patience=5000)
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        scheduler=schedul,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        dataset=dataset_name,
        train_class_stats={"acc_cls": [1], "iu": list(range(n_class))},
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        interval_checkpoint=cfg.get("interval_checkpoint", None),
        interval_weight_update=cfg.get("interval_weight_update", None),
    )
    if not resume:
        shutil.copy(dataset_cfg, osp.join(trainer.out, "polarlys_cfg.txt"))
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
