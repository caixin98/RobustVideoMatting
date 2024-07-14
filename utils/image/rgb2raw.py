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
import random

import cv2 as cv
import numpy as np
import torch
""" Based on http://timothybrooks.com/tech/unprocessing
Functions for forward and inverse camera pipeline. All functions input a torch float tensor of shape (c, h, w).
Additionally, some also support batch operations, i.e. inputs of shape (b, c, h, w)
"""


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]

    num_ccms = len(xyz2cams)
    xyz2cams = torch.tensor(xyz2cams)

    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(0.0, 1.0)
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdims=True)
    return rgb2cam


def random_gains(params=None):
    """Generates random gains for brightening and white balance."""
    # Set default ranges for gains
    rgb_range = (1 / 1.1, 1 / 0.5)
    red_range = (1.9, 2.4)
    blue_range = (1.5, 1.9)

    # Check if params contains the ranges to sample gains from
    if params is not None:
        if "rgb_range" in params:
            rgb_range = params["rgb_range"]
        if "red_range" in params:
            red_range = params["red_range"]
        if "blue_range" in params:
            blue_range = params["blue_range"]

    # RGB gain represents brightening.
    # Sample uniformly in log2 space.
    log2_rgb_gain = random.uniform(math.log2(rgb_range[0]),
                                   math.log2(rgb_range[1]))
    rgb_gain = 2**log2_rgb_gain

    # Red and blue gains represent white balance.
    log2_red_gain = random.uniform(math.log2(red_range[0]),
                                   math.log2(red_range[1]))
    red_gain = 2**log2_red_gain

    log2_blue_gain = random.uniform(math.log2(blue_range[0]),
                                    math.log2(blue_range[1]))
    blue_gain = 2**log2_blue_gain

    return rgb_gain, red_gain, blue_gain


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out


def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = image.clamp(0.0, 1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8)**2.2


def gamma_compression(image):
    """Converts from linear to gammaspace."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8)**(1.0 / 2.2)


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    # no batch dim and CHW tensor
    ccm = ccm.to(image.device).type_as(image)
    if image.dim() == 3 and image.shape[0] == 3:
        shape = image.shape
        image = image.view(3, -1)
        image = torch.mm(ccm, image)
        return image.view(shape)

    # with batch dim and BCHW tensor
    # ccm defined as ccm@color_vec = (colorvec.T @ ccm.T).T
    elif image.dim() == 4 and image.shape[1] == 3:
    
        image = (image.permute([0, 2, 3, 1])[:, :, :, None, :] @ ccm.permute(
            [0, 2, 1])[:, None, None, ...]).squeeze(3).permute([0, 3, 1, 2])
        return image
    else:
        raise RuntimeError(
            f'Got image tensor of shape {image.shape}. Has to be CHW or BCHW with C=3'
        )


def apply_gains(image,
                rgb_gain,
                red_gain,
                blue_gain,
                clip=True,
                apply_rgb_gain=False):
    """Inverts gains while safely handling saturated pixels."""
    if image.dim() == 3:
        image = image[None, ...]
    assert image.dim() == 4 and image.shape[1] in [3, 4]
    batch_size = image.shape[0]
    if image.shape[1] == 3:
        gains = torch.ones([batch_size, 3])
        gains[:, 0] = red_gain
        gains[:, 2] = blue_gain
        if apply_rgb_gain:
            gains *= rgb_gain[:, None]
    else:
        gains = torch.ones([batch_size, 4])
        gains[:, 0] = red_gain
        gains[:, 2] = blue_gain
        if apply_rgb_gain:
            gains *= rgb_gain[:, None]
        # gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]) * rgb_gain
    gains = gains.view(batch_size, -1, 1, 1)
    gains = gains.to(image.device).type_as(image)
    if clip:
        return (image * gains).clamp(0.0, 1.0)
    else:
        return image * gains


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    if image.dim() == 3 and image.shape[0] == 3:
        gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
        gains = gains.view(-1, 1, 1).to(image.device)

        # Prevents dimming of saturated pixels by smoothly masking gains near white.
        gray = image.mean(dim=0, keepdims=True)
        inflection = 0.9
        mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection))**2.0

        safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
        return image * safe_gains
    elif image.dim() == 4 and image.shape[1] == 3:
        gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
        gains = gains.view(1, -1, 1, 1).to(image.device)

        # Prevents dimming of saturated pixels by smoothly masking gains near white.
        gray = image.mean(dim=1, keepdims=True)
        inflection = 0.9
        mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection))**2.0

        safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
        return image * safe_gains
    else:
        raise RuntimeError(
            f'Got image tensor of shape {image.shape}. Has to be CHW or BCHW with C=3'
        )


def mosaic(image, mode='rggb'):
    assert mode in ['rggb', 'grbg', 'bggr']
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if mode == 'rggb':
        red = image[:, 0, 0::2, 0::2]
        green_red = image[:, 1, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 0::2]
        blue = image[:, 2, 1::2, 1::2]
        image = torch.stack((red, green_red, green_blue, blue), dim=1)
    elif mode == 'grbg':
        green_red = image[:, 1, 0::2, 0::2]
        red = image[:, 0, 0::2, 1::2]
        blue = image[:, 2, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 1::2]
        image = torch.stack((green_red, red, blue, green_blue), dim=1)
    elif mode == 'bggr':
        blue = image[:, 0, 0::2, 0::2]
        green_blue = image[:, 1, 0::2, 1::2]
        green_red = image[:, 1, 1::2, 0::2]
        red = image[:, 2, 1::2, 1::2]
        image = torch.stack((blue, green_blue, green_red, red), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))


def demosaic(image, clip=True):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f'image has to be torch.Tensor. got {type(image)} instead.')
    # image = image.clamp(0.0, 1.0) * 255
    # image_normalized = image/image

    if image.dim() == 4:
        num_images = image.shape[0]
        batch_input = True
    else:
        num_images = 1
        batch_input = False
        image = image.unsqueeze(0)

    # Generate single channel input for opencv
    im_sc = torch.zeros(
        (num_images, image.shape[-2] * 2, image.shape[-1] * 2, 1))
    im_sc[:, ::2, ::2, 0] = image[:, 0, :, :]
    im_sc[:, ::2, 1::2, 0] = image[:, 1, :, :]
    im_sc[:, 1::2, ::2, 0] = image[:, 2, :, :]
    im_sc[:, 1::2, 1::2, 0] = image[:, 3, :, :]

    im_sc = im_sc.numpy().astype(np.float32)

    out = []

    for im in im_sc:
        max_v = im.max()
        im_u16 = ((im / max_v) * 255).astype(np.uint8)
        im_dem_np = cv.cvtColor(im_u16, cv.COLOR_BAYER_BG2RGB_VNG)
        im_dem_np = im_dem_np.astype(float) / 255.
        # Convert to torch image
        im_t = _npimage_to_torch(im_dem_np * max_v,
                                 normalize=False,
                                 input_bgr=False)
        if clip:
            im_t = im_t.clamp(0., 1.)
        out.append(im_t)

    if batch_input:
        return torch.stack(out, dim=0)
    else:
        return out[0]


def srgb_to_linear(image):
    rgb2cam = random_ccm()
    cam2rgb = rgb2cam.inverse()
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = invert_smoothstep(image)

    # Inverts gamma compression.
    image = gamma_expansion(image)

    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)

    # Clips saturated pixels.
    image = image.clamp(0.0, 1.0)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def unprocess(image):
    """Unprocesses an image from sRGB to realistic raw data."""

    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = rgb2cam.inverse()
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = invert_smoothstep(image)

    # Inverts gamma compression.
    image = gamma_expansion(image)

    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)

    # Clips saturated pixels.
    image = image.clamp(0.0, 1.0)

    # Applies a Bayer mosaic.
    image = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.012)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    def line(x):
        return 2.18 * x + 1.20

    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def random_noise_levels_unprocessing():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.012)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    def line(x):
        return 2.18 * x + 1.20

    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def random_noise_levels_samsung():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.01)  # 0.01
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    def line(x):
        return 2.18 * x + 1.20

    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)

    return shot_noise, read_noise


def random_noise_levels_high():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.03)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    def line(x):
        return 2.18 * x + 1.20

    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = torch.FloatTensor(image.shape).normal_().to(
        image.device) * variance.sqrt()
    return image + noise


def process_linear_rgb_batch(image, meta_info, return_np=False):
    image = apply_gains(image, meta_info['rgb_gain'], meta_info['red_gain'],
                        meta_info['blue_gain'])
    image = apply_ccm(image, meta_info['cam2rgb'])

    if meta_info['gamma'][0]:
        image = gamma_compression(image)

    if meta_info['smoothstep'][0]:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        image = _torch_to_npimage(image)
    return image


def process_linear_image_raw(image, meta_info, apply_demosaic=True):
    image = apply_gains(image,
                        meta_info['rgb_gain'],
                        meta_info['red_gain'],
                        meta_info['blue_gain'],
                        apply_rgb_gain=False)
    if apply_demosaic:
        image = demosaic(image)
    image = apply_ccm(image, meta_info['cam2rgb'])

    if 'gamma' in meta_info:
        image = gamma_compression(image)

    if 'smoothstep' in meta_info:
        image = apply_smoothstep(image)
    return image.clamp(0.0, 1.0)


def linear_raw_to_rgb(image, meta_info, skip_gamma=False):
    image = apply_gains(image,
                        meta_info['rgb_gain'],
                        meta_info['red_gain'],
                        meta_info['blue_gain'],
                        clip=False,
                        apply_rgb_gain=False)
    image = demosaic(image, clip=False)
    image = apply_ccm(image, meta_info['cam2rgb'])

    if 'gamma' in meta_info and not skip_gamma:
        image = gamma_compression(image)
    return image


def _torch_to_npimage(a: torch.Tensor, unnormalize=True, input_bgr=False):
    a_np = _torch_to_numpy(a.clamp(0.0, 1.0))

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)

    if input_bgr:
        return a_np

    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)


def _npimage_to_torch(a, normalize=True, input_bgr=True):
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    a_t = _numpy_to_torch(a)

    if normalize:
        a_t = a_t / 255.0

    return a_t


def _numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def _torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()



