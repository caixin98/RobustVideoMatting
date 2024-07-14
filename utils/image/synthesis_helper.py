# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of Adobe
# and its suppliers, if any. The intellectual and technical concepts contained
# herein are proprietary to Adobe and its suppliers and are protected by all
# applicable intellectual property laws, including trade secret and copyright
# laws. Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from Adobe.

import copy
import math
import random

import copy
import math
import random

import cv2
import numpy as np
import torch

import torch.nn.functional as F
import gin
from . import rgb2raw


def generate_flow_field(matrix, image_size):
    # 生成一个网格，代表图像中的每个像素的位置
    grid_y, grid_x = torch.meshgrid(torch.arange(0, image_size[0]), torch.arange(0, image_size[1]), indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x).flatten()], dim=0).float()

    # 应用仿射变换矩阵
    new_coords = matrix @ coords
    flow_field = new_coords[:2, :].view(2, image_size[0], image_size[1]) - coords[:2, :].view(2, image_size[0], image_size[1])
    return flow_field

def apply_flow_field(image, flow_field):
    # image shape: [C, H, W]
    # flow_field shape: [2, H, W]，第一个通道为水平位移，第二个通道为垂直位移
    # 获取图像的高度和宽度
    flow_field = flow_field.clone()
    h, w = image.shape[1:]

    # 生成归一化的坐标网格
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
    grid = torch.stack((x, y), 2)  # Shape: [H, W, 2]

    # 调整flow_field的范围从像素位移转为归一化位移
    flow_field = flow_field.permute(1, 2, 0)  # 重排为 [H, W, 2]
    flow_field[..., 0] = flow_field[..., 0] / (w / 2)  # 归一化x
    flow_field[..., 1] = flow_field[..., 1] / (h / 2)  # 归一化y

    # 将光流应用于坐标网格
    new_grid = grid + flow_field

    # 使用grid_sample进行重映射
    # image 需要是 [1, C, H, W] 或 [B, C, H, W]
    # new_grid 需要是 [1, H, W, 2] 或 [B, H, W, 2]
    remapped_image = F.grid_sample(image.unsqueeze(0), new_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)

    return remapped_image.squeeze(0)


def flow_to_color(flow):
    if flow.shape[0] == 2:
        flow = flow.permute(1, 2, 0)
    print(flow.shape)
    # 计算光流的角度和幅度
    magnitude = torch.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = torch.atan2(flow[..., 1], flow[..., 0]) + np.pi
    
    # 归一化角度和幅度
    angle /= 2 * np.pi  # 转换到0-1范围
    normalized_magnitude = torch.clip(magnitude / magnitude.max(), 0, 1)
    
    # 创建HSV表示
    hsv = torch.zeros(flow.shape[0], flow.shape[1], 3)
    hsv[..., 0] = angle
    hsv[..., 1] = 1  # 饱和度固定为最大
    hsv[..., 2] = normalized_magnitude  # 亮度由幅度决定
    
    # HSV到RGB的转换
    return hsv_to_rgb(hsv)

def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    h = (h * 6).int()
    r = torch.zeros_like(h, dtype=torch.float32)
    g = torch.zeros_like(h, dtype=torch.float32)
    b = torch.zeros_like(h, dtype=torch.float32)

    r[(0 <= h) & (h < 1)] = c[(0 <= h) & (h < 1)]
    g[(0 <= h) & (h < 1)] = x[(0 <= h) & (h < 1)]

    r[(1 <= h) & (h < 2)] = x[(1 <= h) & (h < 2)]
    g[(1 <= h) & (h < 2)] = c[(1 <= h) & (h < 2)]

    g[(2 <= h) & (h < 3)] = c[(2 <= h) & (h < 3)]
    b[(2 <= h) & (h < 3)] = x[(2 <= h) & (h < 3)]

    g[(3 <= h) & (h < 4)] = x[(3 <= h) & (h < 4)]
    b[(3 <= h) & (h < 4)] = c[(3 <= h) & (h < 4)]

    r[(4 <= h) & (h < 5)] = x[(4 <= h) & (h < 5)]
    b[(4 <= h) & (h < 5)] = c[(4 <= h) & (h < 5)]

    r[(5 <= h) & (h < 6)] = c[(5 <= h) & (h < 6)]
    b[(5 <= h) & (h < 6)] = x[(5 <= h) & (h < 6)]

    rgb = torch.stack([r, g, b], dim=-1)
    rgb = (rgb + m.unsqueeze(-1)).clip(0, 1)

    return rgb


@gin.configurable(name_or_fn='burst_motion_params')
def get_burst_motion_params(
    max_translation,
    max_rotation,
    max_shear,
    max_scale,
    border_crop,
):
    return locals()


@gin.configurable(name_or_fn='unprocessing_settings')
def get_unprocessing_settings(random_ccm, random_gains, random_rgb_gain_range,
                              smoothstep, gamma, add_noise, noise_type):
    ret = locals()
    ret['random_gains_params'] = {
        # gain to undo when unprocessing the image to raw, e.g.,
        # 1/4.0 for brightening the image by 2 stops when converting to raw.
        'rgb_range': ret.pop('random_rgb_gain_range'),
    }
    return ret


@gin.configurable(name_or_fn='motion_aug_params')
def get_motion_aug_params(size, prob_fgr_affine, prob_bgr_affine, prob_noise,
                          prob_color_jitter, prob_grayscale, prob_sharpness,
                          prob_blur, prob_hflip, prob_pause, static_affine,
                          aspect_ratio_range,motion_affine_params):
    return locals()


def unprocess_images(images, unprocessing_params, randomize_gain):
        if unprocessing_params['random_ccm']:
            rgb2cam = rgb2raw.random_ccm()
        else:
            rgb2cam = torch.eye(3).float()

        if images.dim() == 4:
            rgb2cam = rgb2cam.repeat(images.size(0), 1, 1)

        # Approximately inverts global tone mapping.
        use_smoothstep = unprocessing_params['smoothstep']
        if use_smoothstep:
            images = rgb2raw.invert_smoothstep(images)

        # Inverts gamma compression.
        use_gamma = unprocessing_params['gamma']
        if use_gamma:
            images = rgb2raw.gamma_expansion(images)

        # Inverts color correction.
        images = rgb2raw.apply_ccm(images, rgb2cam)

        # Approximately inverts white balance and brightening.
        if unprocessing_params['random_gains'] and randomize_gain:
            rgb_gain, red_gain, blue_gain = rgb2raw.random_gains(
                unprocessing_params.get('random_gains_params', None))
        else:
            rgb_gain = 1 / .7
            red_gain = 2.1
            blue_gain = 1.7
            # rgb_gain, red_gain, blue_gain = (1.0, 1.0, 1.0)
        images = rgb2raw.safe_invert_gains(images, rgb_gain, red_gain, blue_gain)

        metadata = dict(
            rgb_gain=rgb_gain,
            red_gain=red_gain,
            blue_gain=blue_gain,
            cam2rgb=rgb2cam.inverse(),
            gamma=unprocessing_params['gamma'],
            smoothstep=unprocessing_params['smoothstep'],
        )
        return images, metadata

def apply_noise(images, unprocessing_params):
    if unprocessing_params['add_noise']:
        # All noise model here assume log-read and log-shot parameters are linearly
        # correlated. log_read = 2.18 * log_shot + 1.20
        if unprocessing_params.get('noise_type', 'samsung') == 'samsung':
            shot_noise_level, read_noise_level = rgb2raw.random_noise_levels_samsung(
            )
        elif unprocessing_params['noise_type'] == 'unprocessing':
            shot_noise_level, read_noise_level = rgb2raw.random_noise_levels_unprocessing(
            )
        elif unprocessing_params['noise_type'] == 'high':
            shot_noise_level, read_noise_level = rgb2raw.random_noise_levels_high(
            )
        else:
            raise Exception

        image.clamp_(0.0, 1.0)
        image = rgb2raw.add_noise(images, shot_noise_level,
                                  read_noise_level)
    return images




def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    im_h, im_w = image_shape

    t_trans = np.identity(3)

    t_trans[0, 2] = translation[0]
    t_trans[1, 2] = translation[1]
    print(theta, im_w, im_h)

    t_rot = cv2.getRotationMatrix2D((im_h * 0.5, im_w * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0], [0.0, 0.0, 1.0]])
  
    t_mat = t_scale @ t_rot @ t_shear @ t_trans

    print("t_scale", t_scale)
    print("t_rot", t_rot)
    print("t_shear", t_shear)
    print("t_mat_", t_mat)

    t_mat = t_mat[:2, :]

    return t_mat


def paste_target(fg_image, fg_mask, bg_image, paste_loc):
    valid_mask = np.ones((bg_image.shape[0], bg_image.shape[1]),
                         dtype=np.uint8)
    fg_mask = fg_mask.reshape(fg_mask.shape[0], fg_mask.shape[1], 1)
    x1 = int(paste_loc[0] - 0.5 * fg_image.shape[1])
    x2 = x1 + fg_mask.shape[1]

    y1 = int(paste_loc[1] - 0.5 * fg_image.shape[0])
    y2 = y1 + fg_mask.shape[0]

    x1_pad = max(-x1, 0)
    y1_pad = max(-y1, 0)

    x2_pad = max(x2 - bg_image.shape[1], 0)
    y2_pad = max(y2 - bg_image.shape[0], 0)

    bg_mask = np.zeros((bg_image.shape[0], bg_image.shape[1], 1))

    if x1_pad >= fg_mask.shape[1] or x2_pad >= fg_mask.shape[1] or y1_pad >= fg_mask.shape[0] or y2_pad >= \
        fg_mask.shape[0]:
        return bg_image, valid_mask

    fg_mask_patch = fg_mask[y1_pad:fg_mask.shape[0] - y2_pad,
                            x1_pad:fg_mask.shape[1] - x2_pad, :]

    fg_image_patch = fg_image[y1_pad:fg_mask.shape[0] - y2_pad,
                              x1_pad:fg_mask.shape[1] - x2_pad, :]

    bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = \
      bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] * (1 - fg_mask_patch) \
      + fg_mask_patch * fg_image_patch

    valid_mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 -
               x2_pad] = (fg_mask_patch[:, :, 0] == 0).astype(np.uint8)
    return bg_image, valid_mask


def sample_kernel_sigma(kernel_params):
    sigma_range = kernel_params.get('sigma_range', (0.2, 0.5))
    ksz = kernel_params.get('ksz', 25)

    if kernel_params.get('sigma_sample_space', 'linear') == 'linear':
        ksigma = random.uniform(sigma_range[0], sigma_range[1]) * ksz
    elif kernel_params.get('sigma_sample_space', 'linear') == 'log':
        ksigma = (10**random.uniform(math.log10(sigma_range[0]),
                                     math.log10(sigma_range[1]))) * ksz
    else:
        raise Exception
    return ksigma


def get_standard_kernel(kernel_params):
    ksz = kernel_params.get('ksz', 25)
    if kernel_params['type'] == 'uniform':
        blur_kernel, _ = get_uniform_kernel(ksz=ksz,
                                            radius=kernel_params['k_radius'])

    elif kernel_params['type'] == 'uniform_rand':
        k_radius_range = kernel_params.get('k_radius_range',
                                           (int(ksz * 0.15), int(ksz * 0.4)))
        k_radius = random.randint(k_radius_range[0], k_radius_range[0])
        blur_kernel, _ = get_uniform_kernel(ksz=ksz, radius=k_radius)

    elif 'gauss' in kernel_params['type']:
        theta = 0.0
        if kernel_params['type'] == 'gauss_iso':
            ksigmax = kernel_params['sigma'] * ksz
            ksigmay = ksigmax
        elif kernel_params['type'] == 'gauss_ani':
            ksigmax = kernel_params['sigmax'] * ksz
            ksigmay = kernel_params['sigmay'] * ksz
            theta = kernel_params['theta']
        elif kernel_params['type'] == 'gauss_iso_rand':
            ksigmax = sample_kernel_sigma(kernel_params)
            ksigmay = ksigmax
        elif kernel_params['type'] == 'gauss_ani_rand':
            ksigmax = sample_kernel_sigma(kernel_params)
            ksigmay = sample_kernel_sigma(kernel_params)
            theta = random.uniform(0.0, 360.0)
        else:
            raise Exception

        blur_kernel, _ = get_gaussian_kernel_v2((ksigmax, ksigmay),
                                                ksz=ksz,
                                                theta=theta)

    else:
        raise Exception

    return blur_kernel


def get_kernel_mix_v1(kernel_params):
    p_single = kernel_params.get('p_single', 1.0)
    p_clip = kernel_params.get('p_clip', 0.0)

    kernel_pool = kernel_params.get(
        'kernel_pool', ('uniform_rand', 'gauss_iso_rand', 'gauss_ani_rand'))
    ktype = random.choice(kernel_pool)

    new_kernel_params = copy.deepcopy(kernel_params)
    new_kernel_params['type'] = ktype
    blur_kernel = get_standard_kernel(new_kernel_params)

    if random.random() > p_single:
        ktype2 = random.choice(('uniform_rand', 'gauss_iso_rand'))

        new_kernel_params2 = copy.deepcopy(kernel_params)
        new_kernel_params2['type'] = ktype2
        blur_kernel2 = get_standard_kernel(new_kernel_params2)

        k1_weight = random.random()
        blur_kernel = blur_kernel * k1_weight + \
          blur_kernel2 * (1.0 - k1_weight)
        blur_kernel = blur_kernel / blur_kernel.sum()

    if random.random() < p_clip:
        clip_value = random.uniform(kernel_params.get('min_clip', 0.5),
                                    1.0) * blur_kernel.max()

        blur_kernel[blur_kernel > clip_value] = clip_value
        blur_kernel = blur_kernel / blur_kernel.sum()

    return blur_kernel


def get_blur_kernel(kernel_params):
    ksz = kernel_params.get('ksz', 25)
    if kernel_params is None:
        # Gaussian
        blur_kernel, _ = get_gaussian_kernel_v2((ksz * 0.3, ksz * 0.3),
                                                ksz=ksz)
    elif kernel_params['type'] == 'mixv1':
        blur_kernel = get_kernel_mix_v1(kernel_params)
    else:
        blur_kernel = get_standard_kernel(kernel_params)

    blur_kernel = blur_kernel / blur_kernel.sum()
    return blur_kernel


def get_camera_transformations(downsample_factor, transformation_params):
    """Generate random transformaton parameters.
  translation: uniformly sampled between [-max_translation, max_translation]
      when transformation_params["translation_mode"] is "uniform".
      Elif transformation_params["translation_mode"] is "normal", sampled from
      a Gaussian N(0, translation_sigma)
  rotation: Uniform[-max_rotation, max_rotation].
  shear_x: Uniform[-max_shear, max_shear].
  shear_y: Uniform[-max_shear, max_shear].
  ar_factor (aspect ratio): exp(Uniform[-max_ar_factor, max_ar_factor]).
  scale_factor: exp(Uniform[-max_scale, max_scale]).
  """
    if transformation_params.get('translation_mode', 'uniform') == 'uniform':
        max_translation = transformation_params.get('max_translation', 0.0)

        if max_translation <= 0.01:
            shift = (downsample_factor / 2.0) - 0.5
            translation = (shift, shift)
        else:
            translation = (random.uniform(-max_translation, max_translation),
                           random.uniform(-max_translation, max_translation))
    elif transformation_params.get('translation_mode') == 'normal':
        max_translation = transformation_params.get('max_translation',
                                                    1000000.0)
        translation_sigma = transformation_params.get('translation_sigma')

        translation = (random.gauss(0, translation_sigma) % max_translation,
                       random.gauss(0, translation_sigma) % max_translation)
    else:
        raise Exception

    max_rotation = transformation_params.get('max_rotation', 0.0)
    # theta = random.uniform(-max_rotation, max_rotation)
    theta = max_rotation

    max_shear = transformation_params.get('max_shear', 0.0)
    shear_x = random.uniform(-max_shear, max_shear)
    shear_y = random.uniform(-max_shear, max_shear)
    shear_factor = (shear_x, shear_y)

    max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
    ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

    max_scale = transformation_params.get('max_scale', 0.0)
    scale_factor = np.exp(random.uniform(-max_scale, max_scale))

    scale_factor = (scale_factor, scale_factor * ar_factor)

    return translation, theta, shear_factor, scale_factor


def gauss_2d_rotated(sz, sigma, theta):
    """ Returns a 2-D Gaussian """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor([theta]).float()

    kx = torch.arange(-(sz[0] - 1) / 2,
                      (sz[0] + 1) / 2).reshape(1, -1).to(theta.device)
    ky = torch.arange(-(sz[1] - 1) / 2,
                      (sz[1] + 1) / 2).reshape(-1, 1).to(theta.device)

    # TODO code snippet from https://github.com/sunreef/BlindSR/blob/master/src/degradation.py
    cos_theta = torch.cos(torch.deg2rad(theta))
    sin_theta = torch.sin(torch.deg2rad(theta))

    cos_theta_2 = cos_theta**2
    sin_theta_2 = sin_theta**2

    sigma_x_2 = 2.0 * (sigma[0]**2)
    sigma_y_2 = 2.0 * (sigma[1]**2)

    a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
    b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
    c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

    gauss = torch.exp(-(a * (kx**2) + 2.0 * b * kx * ky + c * (ky**2)))

    return gauss.unsqueeze(0)


def get_gaussian_kernel_v2(sd, ksz, theta=0.0):
    """ Returns a Gaussian kernel with standard deviation sd """
    assert ksz % 2 == 1
    K = gauss_2d_rotated(ksz, sd, theta)
    K = K / K.sum()
    return K.unsqueeze(0), ksz


def get_uniform_kernel(ksz, radius):
    """ Returns a Gaussian kernel with standard deviation sd """
    assert ksz % 2 == 1
    K = torch.zeros((1, 1, ksz, ksz))

    pad = (ksz - 1) // 2 - radius
    K[:, :, pad:-pad, pad:-pad] = 1.0
    K = K / K.sum()
    return K, ksz


