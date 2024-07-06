# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of Adobe
# and its suppliers, if any. The intellectual and technical concepts contained
# herein are proprietary to Adobe and its suppliers and are protected by all
# applicable intellectual property laws, including trade secret and copyright
# laws. Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from Adobe.

import math
from math import exp

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

# def avg_demosiac(image_tensor):
# return torch.stack([image_tensor[:,0,:,:], (image_tensor[:,1,:,:]+image_tensor[:,2,:,:])/2, image_tensor[:, 3, :, :]], dim=1)


@gin.configurable(module='image_loss')
class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
    calculation """

    def __init__(self, metric='l1', boundary_ignore=None, tone_curve=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        # self.apply_finishing = apply_finishing
        tone_curve = tone_curve
        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':

            def l2_sqrt(pred, gt):
                return (((pred - gt)**2).sum(dim=-3)).sqrt().mean()

            self.loss_fn = l2_sqrt
        elif metric == 'bce_logits':
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif metric == 'charbonnier':

            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt)**2 + eps**2).sqrt().mean()

            self.loss_fn = charbonnier
        else:
            raise ValueError(f'Unknown metric: {metric}')

        if tone_curve is None:
            self.tone_curve = nn.Identity()
        else:
            self.tone_curve = tone_curve

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore,
                    self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                              self.boundary_ignore:-self.boundary_ignore]
        pred = self.tone_curve(pred)
        gt = self.tone_curve(gt)
        if valid is None:
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (
                valid.float().sum() * elem_ratio + eps)

            # err = (err * valid.float()).sum() / (valid.float().sum() + eps)

        return err


class PixelWiseErrorwTextureMask(nn.Module):

    def __init__(self,
                 metric='l1',
                 boundary_ignore=None,
                 win_size=4,
                 var_thresh=None):
        super().__init__()
        self.loss = PixelWiseError(metric=metric,
                                   boundary_ignore=boundary_ignore)
        self.win_size = win_size
        self.var_thresh = var_thresh

    def forward(self, pred, gt, ref_im, valid=None):
        assert ref_im.shape[-1] % self.win_size == 0

        with torch.no_grad():
            ref_mean = F.avg_pool2d(ref_im,
                                    self.win_size,
                                    stride=self.win_size,
                                    padding=0)
            ref_mean_re = F.interpolate(ref_mean,
                                        scale_factor=self.win_size,
                                        mode='nearest')
            ref_im_demean = ref_im - ref_mean_re
            err = ref_im_demean**2
            err = F.avg_pool2d(err,
                               self.win_size,
                               stride=self.win_size,
                               padding=0)
            valid_mask = (err > self.var_thresh).float()
            valid_mask = F.interpolate(valid_mask,
                                       size=pred.shape[-2:],
                                       mode='nearest')

        if valid is None:
            valid = valid_mask
        else:
            valid = valid * valid_mask

        l = self.loss(pred, gt, valid)
        return l


class GradError(nn.Module):

    def __init__(self, metric='l1', boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore,
                    self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                              self.boundary_ignore:-self.boundary_ignore]

        pred_dx = pred[..., :-1] - pred[..., 1:]
        pred_dy = pred[..., :-1, :] - pred[..., 1:, :]

        gt_dx = gt[..., :-1] - gt[..., 1:]
        gt_dy = gt[..., :-1, :] - gt[..., 1:, :]

        if valid is None:
            err = self.loss_fn(pred_dx, gt_dx) + self.loss_fn(pred_dy, gt_dy)
        else:
            err = self.loss_fn(pred_dx, gt_dx,
                               reduction='none') + self.loss_fn(
                                   pred_dy, gt_dy, reduction='none')

            eps = 1e-12
            err = (err * valid.float()).sum() / (valid.float().sum() + eps)

        err = err * 0.5
        return err


@gin.configurable(module='image_loss')
class PSNR(nn.Module):

    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value',
                                           1.0)) - 10.0 * mse.log10()
        else:
            psnr = 20 * gt.max().log10() - 10.0 * mse.log10()

        if torch.isinf(psnr) or torch.isnan(psnr):
            # print('invalid psnr')
            psnr = torch.zeros_like(psnr)
        return psnr

    def forward(self, pred, gt, valid=None):
        if valid is None:
            psnr_all = [
                self.psnr(p.unsqueeze(0), g.unsqueeze(0))
                for p, g in zip(pred, gt)
            ]
        else:
            psnr_all = [
                self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0))
                for p, g, v in zip(pred, gt, valid)
            ]

        psnr_all = [
            p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))
        ]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class SSIM(nn.Module):

    def __init__(self, boundary_ignore=None, use_for_loss=True):
        super().__init__()
        self.ssim = SSIMModule(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore,
                        self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore,
                    self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore,
                              self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (
                valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):

    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        raise NotImplementedError('lpips not supported right now.')
        # super().__init__()
        # self.boundary_ignore = boundary_ignore
        # self.bgr2rgb = bgr2rgb

        # if type == 'alex':
        #     self.loss = lpips.LPIPS(net='alex')
        # elif type == 'vgg':
        #     self.loss = lpips.LPIPS(net='vgg')
        # else:
        #     raise Exception

    def forward(self, pred, gt, valid=None):
        raise NotImplementedError('lpips not supported right now.')
        # if self.bgr2rgb:
        #     pred = pred[..., [2, 1, 0], :, :].contiguous()
        #     gt = gt[..., [2, 1, 0], :, :].contiguous()

        # if self.boundary_ignore is not None:
        #     pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
        #     gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # loss = self.loss(pred, gt)

        # return loss.mean()


class MappedLoss(nn.Module):

    def __init__(self, base_loss, mapping_fn=None):
        super().__init__()
        self.base_loss = base_loss
        self.mapping_fn = mapping_fn

    def forward(self, pred, gt, meta_info=None, valid=None):
        if self.mapping_fn is not None:
            pred_l = [self.mapping_fn(p, m) for p, m in zip(pred, meta_info)]
            gt_l = [self.mapping_fn(p, m) for p, m in zip(gt, meta_info)]
            pred = torch.stack(pred_l)
            gt = torch.stack(gt_l)

        err = self.base_loss(pred, gt, valid)
        return err


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size, channel=1):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()
    return window


def _ssim(img1,
          img2,
          window_size=11,
          window=None,
          size_average=True,
          full=False,
          val_range=None,
          spatial_out=False):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = _create_window(real_size, channel=channel).to(img1.device)

    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if spatial_out:
        ret = ssim_map
    elif size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
#     device = img1.device
#     weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
#     levels = weights.size()[0]
#     mssim = []
#     mcs = []
#     for _ in range(levels):
#         sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
#         mssim.append(sim)
#         mcs.append(cs)

#         img1 = F.avg_pool2d(img1, (2, 2))
#         img2 = F.avg_pool2d(img2, (2, 2))

#     mssim = torch.stack(mssim)
#     mcs = torch.stack(mcs)

#     # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
#     if normalize:
#         mssim = (mssim + 1) / 2
#         mcs = (mcs + 1) / 2

#     pow1 = mcs ** weights
#     pow2 = mssim ** weights
#     # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
#     output = torch.prod(pow1[:-1] * pow2[-1])
#     return output


# Classes to re-use window
class SSIMModule(torch.nn.Module):

    def __init__(self,
                 window_size=11,
                 size_average=True,
                 val_range=None,
                 spatial_out=False):
        super(SSIMModule, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.spatial_out = spatial_out

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = _create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = _create_window(self.window_size,
                                    channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        window = window.to(img1.device)
        return _ssim(img1,
                     img2,
                     window=window,
                     window_size=self.window_size,
                     size_average=self.size_average,
                     spatial_out=self.spatial_out)
