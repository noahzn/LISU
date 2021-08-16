"""
@FileName: utils.py
@Time    : 5/1/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import os
import torch
import shutil
import numpy as np
import math

import matplotlib.pyplot as plt
from PIL import Image
import math

cmap = plt.cm.viridis


def save_checkpoint(params, is_best=False, directory='/data2/zhangn/project/checkpoint/', filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{0}{1}'.format(directory, filename)
    torch.save(params, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image
    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    image = image.permute(0, 2, 3, 1)
    x = torch.argmax(image, dim=-1)

    return x

def colorize(x):
    colour_code = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])

    # colour_code = np.uint8(colour_code * 255)
    x = colour_code[x.detach().cpu().numpy().astype(int)]

    return torch.from_numpy(x).permute(0, 1, 2)


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr

    return lr

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(
        n_class, n_class)

    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc, iu


class DepthMetrics:
    def __init__(self):
        self.num = 0
        self.threshold_1_25 = 0
        self.threshold_1_25_2 = 0
        self.threshold_1_25_3 = 0
        self.rmse_linear = 0.0
        self.rmse_log = 0.0
        self.rmse_log_scale_invariant = 0.0
        self.ard = 0.0
        self.srd = 0.0

    def reset(self):
        self.__init__()

    def compute(self, d, d_gt):
        self.num += 1
        input_gt_depth_image = d_gt.data.squeeze().cpu().numpy().astype(np.float32)
        pred_depth_image = d.data.squeeze().cpu().numpy().astype(np.float32)

        input_gt_depth_image /= np.max(input_gt_depth_image)
        pred_depth_image /= np.max(pred_depth_image)

        n = np.sum(input_gt_depth_image > 1e-3)  # 计算值大于1e-3的个数

        idxs = (input_gt_depth_image <= 1e-3)  # 返回与原始数据同维的布尔值
        pred_depth_image[idxs] = 1  # 将小于1e-3赋值成1
        input_gt_depth_image[idxs] = 1

        pred_d_gt = pred_depth_image / input_gt_depth_image
        pred_d_gt[idxs] = 100
        gt_d_pred = input_gt_depth_image / (pred_depth_image+0.0001)
        gt_d_pred[idxs] = 100

        self.threshold_1_25 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n  # np.maximum返回相对较大的值
        self.threshold_1_25_2 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
        self.threshold_1_25_3 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n

        log_pred = np.log(pred_depth_image+0.0001)
        log_gt = np.log(input_gt_depth_image)

        d_i = log_gt - log_pred

        self.rmse_linear += np.sqrt(np.sum((pred_depth_image - input_gt_depth_image) ** 2) / n)
        self.rmse_log += np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)
        self.rmse_log_scale_invariant += np.sum(d_i ** 2) / n + (np.sum(d_i) ** 2) / (n ** 2)
        self.ard += np.sum(np.abs((pred_depth_image - input_gt_depth_image)) / input_gt_depth_image) / n
        self.srd += np.sum(((pred_depth_image - input_gt_depth_image) ** 2) / input_gt_depth_image) / n

    def get_results(self):
        self.threshold_1_25 /= self.num
        self.threshold_1_25_2 /= self.num
        self.threshold_1_25_3 /= self.num
        self.rmse_linear /= self.num
        self.rmse_log /= self.num
        self.rmse_log_scale_invariant /= self.num
        self.ard /= self.num
        self.srd /= self.num

        return [self.threshold_1_25, self.threshold_1_25_2, self.threshold_1_25_3, self.rmse_linear, self.rmse_log, self.rmse_log_scale_invariant, self.ard, self.srd]


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = ((target > 0) + (output > 0)) > 0

        output = 1e3 * output[valid_mask]
        target = 1e3 * target[valid_mask]
        abs_diff = (output - target).abs()
        # input(abs_diff.size())
        self.mse = float((torch.pow(abs_diff, 2)).mean())

        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        # self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.lg10 = (log10(output) - log10(target)).abs()
        self.lg10 = self.lg10[~torch.isinf(self.lg10)]
        self.lg10 = float(self.lg10[~torch.isnan(self.lg10)].mean())

        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge

def merge_into_row_with_r(input, r, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    r = 255 * np.transpose(np.squeeze(r.cpu().numpy()), (1, 2, 0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, r, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)