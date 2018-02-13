import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn
import traceback
import logging
import time


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        logging.basicConfig(filename="trainer.log")
        self.logger = logging.getLogger()

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []

        crash_batch_idx = -2

        hist = np.zeros((n_class, n_class))
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            try:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                score = self.model(data)
                try:
                    loss = cross_entropy2d(score, target, weight=torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda(),
                                       size_average=self.size_average)
                except ValueError:
                    filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0]
                    self.logger.warning("Whole label for {} is unlabeled (-1)".format(filename))
                    del target, data
                    if "loss" in locals():
                        del loss
                    if "score" in locals():
                        del score
                    if "lbl_pred" in locals():
                        del lbl_pred
                    if "lbl_true" in locals():
                        del lbl_true
                    continue
                if np.isnan(float(loss.data[0])):
                    del loss, data, target, score
                    filename = self.val_loader.dataset.files["valid"][batch_idx]["data"][0].split("/")
                    self.logger.warning("Loss was NaN while validating\n:image {}".format(filename))
                val_loss += float(loss.data[0]) / len(data)

                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()

                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    if len(visualizations) < 9:
                        img, lt = self.val_loader.dataset.untransform(img, lt)
                        if len(img.shape) == 2:
                            img = np.repeat(img[:, :, np.newaxis], 3, 2)
                        viz = fcn.utils.visualize_segmentation(
                            lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                        visualizations.append(viz)
                        hist += torchfcn.utils.fast_hist(lt.flatten(), lp.flatten(), n_class)
                    else:
                        hist += torchfcn.utils.fast_hist(lt.numpy().flatten(), lp.flatten(), n_class)
            except RuntimeError:
                del target, data
                if "score" in locals():
                    del score
                if "loss" in locals():
                    del loss
                filename = self.val_loader.dataset.files[batch_idx]["data"][0]
                self.logger.exception("Out of memory while validating:\nimage: {}".format(filename))
                if crash_batch_idx == batch_idx:
                    self.val_loader.dataset.set_data_ranges(len(self.val_loader.dataset.data_ranges), 0)  # TODO, this changes size of files list
                    self.val_loader.dataset.reload_files_from_default_index()
                else:
                    print("Sleeping for one hour and hope it resolves itself")
                    time.sleep(1 * 5)  # sleep for one hour

                crash_batch_idx = batch_idx

        metrics = torchfcn.utils.label_accuracy_score_from_hist(hist)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                (datetime.datetime.now(pytz.timezone('Europe/Oslo')) - \
                self.timestamp_start).total_seconds()

            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'out': self.out,
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        crash_batch_idx = -2

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration != 0:
                self.validate()

            assert self.model.training

            try:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optim.zero_grad()
                score = self.model(data)

                try:
                    loss = cross_entropy2d(score, target, weight=torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda(),
                                       size_average=self.size_average)
                except ValueError:
                    filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0]
                    self.logger.warning("Whole label for {} is unlabeled (-1)".format(filename))
                    del target, data
                    if "loss" in locals():
                        del loss
                    if "score" in locals():
                        del score
                    if "lbl_pred" in locals():
                        del lbl_pred
                    if "lbl_true" in locals():
                        del lbl_true
                    continue
                loss /= len(data)  # average loss over batch
                if np.isnan(float(loss.data[0])):
                    del loss, data, target, score
                    filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0].split("/")
                    self.logger.warning("Loss was NaN while training\n:image {}".format(filename))
                    continue  # likely could not load image from nas, just continue
                loss.backward()
                self.optim.step()

                metrics = []
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu().numpy()
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        lbl_true, lbl_pred, n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
                metrics = np.mean(metrics, axis=0)

                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    elapsed_time = (datetime.datetime.now(pytz.timezone('Europe/Oslo')) - self.timestamp_start).total_seconds()

                    log = [self.epoch, self.iteration] + [loss.data[0]] + \
                        metrics.tolist() + [''] * 5 + [elapsed_time]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')

                if self.iteration >= self.max_iter:
                    break

            except RuntimeError:  # assumed always out of memory?
                del target, data
                if "score" in locals():
                    del score
                if "loss" in locals():
                    del loss
                filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0]
                self.logger.exception("Out of memory while training:\nimage: {}".format(filename))
                if crash_batch_idx == batch_idx - 1:
                    self.train_loader.dataset.set_data_ranges(len(self.train_loader.dataset.data_ranges), 0)
                    self.train_loader.dataset.reload_files_from_default_index()
                else:
                    print("Sleeping for one hour and hope it resolves itself")
                    time.sleep(1 * 5)  # sleep for one hour

                crash_batch_idx = batch_idx

    def train(self):
        try:
            max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
            for epoch in tqdm.trange(self.epoch, max_epoch,
                                     desc='Train', ncols=80):

                self.epoch = epoch
                self.train_epoch()
                if self.iteration >= self.max_iter:
                    break
        except Exception as e:
            self.logger.exception("Unexpected error")