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
import sys


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
                 train_loader, val_loader, out, max_iter, scheduler=None, dataset=None, train_class_stats=None, shuffle=True,
                 size_average=False, interval_validate=None, interval_checkpoint=None, interval_weight_update=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.dataset = dataset

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Europe/Oslo'))
        self.size_average = size_average

        if shuffle:
            train_loader_shuffle = getattr(self.train_loader.dataset, "shuffle_files", None)
            self.shuffle = train_loader_shuffle is not None and callable(train_loader_shuffle)

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.interval_checkpoint = interval_checkpoint
        self.interval_weight_update = interval_weight_update

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.n_class = len(self.val_loader.dataset.class_names)
        class_names = self.val_loader.dataset.class_names

        self.log_headers = [
            'epoch',
            'iteration',
            'filename',
            'train/loss',
            'train/acc',
            'train/mean_acc_cls',
            'train/fwavacc',
            'train/mean_iu',
            'valid/loss',
            'valid/acc',
            'valid/mean_acc_cls',
            'valid/fwavacc',
            'valid/mean_iu',
            'valid/mean_bj',
            'elapsed_time',
        ]

        self.aux = getattr(model, "use_aux", False)

        if self.aux:
            self.log_headers.insert(self.log_headers.index("train/loss") + 1, "train/loss_aux")

        self.train_class_stats = train_class_stats
        if train_class_stats is not None:
            for metric, class_idxs in train_class_stats.items():
                idx = self.log_headers.index("train/mean_{}".format(metric))
                class_headers = ["train/{}_{}".format(metric, class_name) for i, class_name in enumerate(class_names) if i in class_idxs]
                self.log_headers[idx:idx] = class_headers

        for metric in ["valid/mean_acc_cls", "valid/mean_iu", "valid/mean_bj"]:
            metric_index = self.log_headers.index(metric)
            if "mean_" in metric:
                metric = metric.replace("mean_", "")
            class_headers = ["{}_{}".format(metric, class_name) for class_name in class_names]
            self.log_headers[metric_index:metric_index] = class_headers

        self.valid_headers_count = sum(1 if "valid" in header else 0 for header in self.log_headers)
        self.train_headers_count = sum(1 if "train" in header else 0 for header in self.log_headers)

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        logging.basicConfig(filename="trainer.log", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger()

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_bj = 0

        self.metadata = getattr(model, "metadata", False)

    def validate(self):
        def free_memory(variables):
            if "batch" in variables:
                nonlocal batch
                del batch
            if "target" in variables:
                nonlocal target
                del target
            if "data" in variables:
                nonlocal data
                del data
            if "loss" in variables:
                nonlocal loss
                del loss
            if "score" in variables:
                nonlocal score
                del score
            if "lbl_pred" in variables:
                nonlocal lbl_pred
                del lbl_pred
            if "lbl_true" in variables:
                nonlocal lbl_true
                del lbl_true
            torch.cuda.empty_cache()
        training = self.model.training
        self.model.eval()

        val_loss = 0
        visualizations = []

        crash_batch_idx = -2

        metrics = np.zeros((len(self.val_loader), self.valid_headers_count))
        hist = np.zeros((self.n_class, self.n_class))
        for batch_idx, batch in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):

            data, target = batch[0].cuda(), batch[1].cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.metadata:
                metadata = batch[2].cuda()
                metadata = Variable(metadata, volatile=True)
                score = self.model(data, metadata)
            else:
                score = self.model(data)

            try:
                loss = cross_entropy2d(score, target, weight=torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda(),
                                   size_average=self.size_average)
            except ValueError:
                free_memory(locals())
                filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0]
                self.logger.warning("Whole label for {} is unlabeled (-1)".format(filename))
                continue
            if np.isnan(float(loss.data[0])):
                free_memory(locals())
                filename = self.val_loader.dataset.files["valid"][batch_idx]["data"][0].split("/")
                self.logger.warning("Loss was NaN while validating\n:image {}".format(filename))
                continue

            val_loss = float(loss.data[0]) / len(data)


            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            if len(visualizations) < 9:
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, 2)
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class)
                    visualizations.append(viz)

            batch_metrics = [val_loss]
            batch_metrics.extend(list(self.flatten(torchfcn.utils.label_accuracy_score(lbl_true.numpy(), lbl_pred, self.n_class))))
            batch_metrics.extend(torchfcn.utils.boundary_jaccard(lbl_true.numpy(), lbl_pred, range(self.n_class)))
            metrics[batch_idx, :] = batch_metrics

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                (datetime.datetime.now(pytz.timezone('Europe/Oslo')) - \
                 self.timestamp_start).total_seconds()

            for batch_idx in range(metrics.shape[0]):
                log = [self.epoch, self.iteration, self.val_loader.dataset.get_filename(batch_idx, with_radar=True)] + [''] * self.train_headers_count + \
                      list(metrics[batch_idx, :]) + [""]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            log = [self.epoch, self.iteration, ""] + [''] * self.train_headers_count + \
                    list(np.nanmean(metrics, axis=0)) + [elapsed_time]

            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_bj = np.nanmean(metrics[:, -1])
        is_best = mean_bj > self.best_mean_bj
        if is_best:
            self.best_mean_bj = mean_bj
        torch.save({
            'out': self.out,
            'dataset': self.dataset,
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_bj': self.best_mean_bj,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

        free_memory(locals())



    def train_epoch(self):
        def free_memory(variables):
            if "target" in variables:
                nonlocal target
                del target
            if "data" in variables:
                nonlocal data
                del data
            if "loss" in variables:
                nonlocal loss
                del loss
            if "score" in variables:
                nonlocal score
                del score
            if "lbl_pred" in variables:
                nonlocal lbl_pred
                del lbl_pred
            if "lbl_true" in variables:
                nonlocal lbl_true
                del lbl_true
            torch.cuda.empty_cache()
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        crash_batch_idx = -2

        self.optim.zero_grad()

        if self.shuffle:
            self.train_loader.dataset.shuffle_files(self.epoch)

        start_idx = self.iteration % len(self.train_loader)
        if start_idx != 0:
            print("Resuming from checkpoint during epoch, starting at iteration {}".format(start_idx))

        for batch_idx, batch in tqdm.tqdm(enumerate(self.train_loader.iter_from(start_idx)),
                                          total=len(self.train_loader) - start_idx,
                                          desc='Train epoch=%d' % self.epoch, ncols=80,
                                          leave=False):

            if self.iteration % self.interval_validate == 0 and self.iteration != 0:
                self.validate()

            assert self.model.training

            data, target = batch[0].cuda(), batch[1].cuda()
            data, target = Variable(data), Variable(target)
            batch_size = len(data)

            if self.metadata:
                metadata = batch[2].cuda()
                metadata = Variable(metadata)
                if self.aux:
                    score, aux = self.model(data, metadata)
                else:
                    score = self.model(data, metadata)
            else:
                if self.aux:
                    score, aux = self.model(data)
                else:
                    score = self.model(data)

            del batch

            try:
                loss = cross_entropy2d(score, target, weight=torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda(),
                                   size_average=self.size_average)
                if self.aux:
                    aux_loss = cross_entropy2d(aux, target, weight=torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda(),
                                   size_average=self.size_average)
            except ValueError:
                free_memory(locals())
                filename = self.train_loader.dataset.files["train"][batch_idx]["data"][0]
                self.logger.warning("Whole label for {} is unlabeled (-1)".format(filename))
                continue

            if self.aux:
                loss = loss + 0.4 * aux_loss

            loss /= batch_size  # average loss over batch

            if np.isnan(float(loss.data[0])):
                print("nan loss")
                free_memory(locals())
                self.logger.warning("Loss was NaN while training\n:image {}".format(self.train_loader.dataset.get_filename(batch_idx)))
                continue  # likely could not load image from nas, just continue
            loss.backward()

            if self.interval_weight_update is None or self.iteration % self.interval_weight_update == 0 and self.iteration != 0:
                self.optim.step()
                self.optim.zero_grad()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            metrics = torchfcn.utils.label_accuracy_score(lbl_true, lbl_pred, n_class=n_class, per_class=self.train_class_stats is not None)
            #metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('Europe/Oslo')) - self.timestamp_start).total_seconds()

                log = [self.epoch, self.iteration, self.train_loader.dataset.get_filename(batch_idx, with_radar=True)] + [loss.data[0]]

                if self.aux:
                    log.append(aux_loss.data[0])

                if self.train_class_stats is not None:
                    log.append(metrics[0])
                    if "acc_cls" in self.train_class_stats:
                        log.extend([c_m for i, c_m in enumerate(metrics[1]) if i in self.train_class_stats["acc_cls"]])
                    log.extend(metrics[2:4])
                    if "iu" in self.train_class_stats:
                        log.extend([c_m for i, c_m in enumerate(metrics[4]) if i in self.train_class_stats["iu"]])
                    log.append(metrics[5])
                else:
                    log.extend(list(metrics))
                log.extend([''] * self.valid_headers_count + [elapsed_time])
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.scheduler is not None:
                self.scheduler.step(loss.data[0])

            self.iteration += 1

            if self.interval_checkpoint is not None and self.iteration % self.interval_checkpoint == 0 and self.iteration != 0 and batch_idx != 0:
                try:
                    torch.save({
                        'out': self.out,
                        'dataset': self.dataset,
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'best_mean_bj': self.best_mean_bj,
                    }, osp.join(self.out, 'checkpoint.pth.tar'))
                    print("Successfully saved checkpoint at iteration {}".format(self.iteration))
                except Exception as e:
                    self.logger.exception("Could not save checkpoint")
                    print("Could not save checkpoint")

            if self.iteration >= self.max_iter:
                break

    def train(self):
        try:
            max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
            for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
                self.epoch = epoch
                self.train_epoch()
                if self.iteration >= self.max_iter:
                    break
        except RuntimeError as e:
            print(e)
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            self.logger.exception("Runtime Error, likely out of memory.")
            self.restart_script()
        except Exception as e:
            print(e)
            traceback.print_stack()
            print('--------------')
            traceback.print_exc()
            print('--------------')
            self.logger.exception("Unexpected error")
            self.restart_script()

    def restart_script(self, resume=True):
        print("RESTARTING SCRIPT")
        torch.cuda.empty_cache()
        script_name = sys.argv[0]
        arguments = ["python", script_name]
        if len(sys.argv) > 1:
            arguments.extend(sys.argv[1:])
        if resume and not any("--resume" in argument for argument in arguments):
            arguments.extend(["--resume", osp.join(self.out, "checkpoint.pth.tar")])
        os.execv(sys.executable, arguments)

    def flatten(self, l):
        for el in l:
            try:
                yield from self.flatten(el)
            except TypeError:
                yield el

