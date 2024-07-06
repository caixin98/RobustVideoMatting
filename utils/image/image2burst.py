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
import pdb
import random

import cv2
import gin
import torch
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F

from .augmentation import center_crop
from .augmentation import random_resized_crop
from . import rgb2raw
from utils.image.synthesis_helper import get_camera_transformations,get_tmat


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


def unprocess_image(image, unprocessing_params, randomize_gain):
        if unprocessing_params['random_ccm']:
            rgb2cam = rgb2raw.random_ccm()
        else:
            rgb2cam = torch.eye(3).float()

        # Approximately inverts global tone mapping.
        use_smoothstep = unprocessing_params['smoothstep']
        if use_smoothstep:
            image = rgb2raw.invert_smoothstep(image)

        # Inverts gamma compression.
        use_gamma = unprocessing_params['gamma']
        if use_gamma:
            image = rgb2raw.gamma_expansion(image)

        # Inverts color correction.
        image = rgb2raw.apply_ccm(image, rgb2cam)

        # Approximately inverts white balance and brightening.
        if unprocessing_params['random_gains'] and randomize_gain:
            rgb_gain, red_gain, blue_gain = rgb2raw.random_gains(
                unprocessing_params.get('random_gains_params', None))
        else:
            rgb_gain = 1 / .7
            red_gain = 2.1
            blue_gain = 1.7
            # rgb_gain, red_gain, blue_gain = (1.0, 1.0, 1.0)
        image = rgb2raw.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)

        metadata = dict(
            rgb_gain=rgb_gain,
            red_gain=red_gain,
            blue_gain=blue_gain,
            cam2rgb=rgb2cam.inverse(),
            gamma=unprocessing_params['gamma'],
            smoothstep=unprocessing_params['smoothstep'],
        )
        return image, metadata

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

@gin.configurable()
class GenerateBurst:
    """
    This class generates artificial image bursts with simulated motion blur and noise.

    Args:
        crop_sz (int, optional): Size of the random crop applied to the image. Defaults to None.
        burst_size (int, optional): Number of images in the generated burst. Defaults to 8.
        downsample_factor (float, optional): Factor to downsample the image. Defaults to 1.
        cfa_pattern (numpy.ndarray, optional): Bayer filter pattern for color aliasing simulation. Defaults to None.
        alpha (tuple, optional): Parameters for elastic deformation noise. Defaults to (10.0, 10.0).
        sigma (tuple, optional): Parameters for elastic deformation noise. Defaults to (2.0, 2.0).
        transform (callable, optional): A function to apply additional transformations to the image. Defaults to None.
        crop_scale_range (tuple, optional): Range for random crop scale. Defaults to None.
        crop_ar_range (tuple, optional): Range for random crop aspect ratio. Defaults to None.
        random_crop (bool, optional): Whether to apply random cropping. Defaults to True.
        image_key (str, optional): Key name for the image data in the input dictionary. Defaults to 'image'.
    """
    def __init__(self,
                 crop_sz = None,
                 burst_size = 8,
                 downsample_factor = 1,
                 cfa_pattern = None,
                 alpha = (10.0, 10.0),
                 sigma = (2.0, 2.0),
                 transform=None,
                 crop_scale_range=None,
                 crop_ar_range=None,
                 random_crop=False,
                 image_key='image'):

        self.crop_sz = crop_sz
        self.image_key = image_key
        self.burst_size = burst_size
        self.downsample_factor = downsample_factor

        self.burst_transformation_params = get_burst_motion_params()
        self.unprocessing_params = get_unprocessing_settings()

        self.crop_scale_range = crop_scale_range
        self.crop_ar_range = crop_ar_range
        self.random_crop = random_crop

        self.transform = transform
        self.cfa_pattern = cfa_pattern
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, data, movement=True):

        img_key = self.image_key
        if not isinstance(data, dict):
            data = {self.image_key: data}
        # Augmentation, e.g. convert to tensor
        if self.transform is not None:
            data[img_key] = self.transform(data[img_key])

        # Random crop
        if self.crop_sz is not None:
            if getattr(self, 'random_crop', True):
                frame_crop = random_resized_crop(
                    data[img_key],
                    self.crop_sz,
                    scale_range=self.crop_scale_range,
                    ar_range=self.crop_ar_range)
            else:
                assert self.crop_scale_range is None and self.crop_ar_range is None
                frame_crop = center_crop(data[img_key], self.crop_sz)
        else:
            frame_crop = data[img_key]

        # Generate a burst of raw images from the processed frame
        burst_rgb, frame_gt, metadata = self.rgb2rawburst(image=frame_crop, movement=movement)

        # Apply CFA pattern if specified
        if self.cfa_pattern:
            burst_rgb = rgb2raw.mosaic(burst_rgb, self.cfa_pattern)

        return burst_rgb, frame_gt, metadata
    
    def rgb2rawsingle(self, image, randomize_gain=True, add_noise=True):
        """Convert an RGB image to a 'raw' image simulating camera processing."""
        unprocessing_params = self.unprocessing_params
        image, metadata = unprocess_image(image, unprocessing_params, randomize_gain)
        # Apply noise if specified
        if add_noise:
            image = apply_noise(image, unprocessing_params)
        return image.clamp(0.0, 1.0), metadata



    def rgb2rawburst(self, image: torch.Tensor, movement: bool = True):
        burst_size = self.burst_size
        unprocessing_params = self.unprocessing_params
        # unprocess without adding noise
        raw, metadata = self.rgb2rawsingle(image, add_noise=False)
        # Generate burst
        if movement:
            raw = raw.permute(1, 2, 0).numpy()
            image_burst_rgb, image_gt, shifts = self.raw2burst(raw)
            image_burst_rgb = [
                torch.from_numpy(image_burst_rgb[i]).float().permute(2, 0, 1) *
                1 for i in range(burst_size)
            ]
            image_burst_rgb = torch.stack(image_burst_rgb, dim=0)
            image_gt = torch.from_numpy(image_gt).float().permute(2, 0, 1)
        else:
            shifts = None
            image_burst_rgb = raw.repeat(burst_size, 1, 1, 1)
            image_gt = image_burst_rgb[0]

        if unprocessing_params['add_noise']:
            apply_noise(image_burst_rgb, unprocessing_params)

        metadata['shifts'] = shifts
        return image_burst_rgb.clamp(0.0, 1.0), image_gt, metadata

    
    def raw2burst(self, image):
        """Generates a burst using random affine transformations
        including translation, rotation, shearing and scaling.
        Also adds elastic deformation with 40% probability.
        """
        burst_size = self.burst_size
        downsample_factor = self.downsample_factor
        transformation_params = self.burst_transformation_params
        interpolation = cv2.INTER_LINEAR

        burst = []
        if burst_size == 1:
            return [image], image, None

        shifts = []
        shifts_inv = []
        cvs, rvs = torch.meshgrid(
            [torch.arange(0, image.shape[0]),
             torch.arange(0, image.shape[1])],
            indexing='xy')

        sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)),
                                  dim=-1).float()
        image_gt = copy.deepcopy(image)

        # Sample object
        for i in range(burst_size):
            if i == 0:
                shift = 0.0
                translation = (shift, shift)
                theta = 0.0
                shear_factor = (0.0, 0.0)
                scale_factor = (1.0, 1.0)
            else:
                translation, theta, shear_factor, scale_factor = get_camera_transformations(
                    downsample_factor, transformation_params)

            output_sz = (image.shape[1], image.shape[0])

            t_mat = get_tmat((image.shape[0], image.shape[1]), translation,
                             theta, shear_factor, scale_factor)
            t_mat_tensor = torch.from_numpy(t_mat)

            if i == 0:
                image_t = cv2.warpAffine(image_gt,
                                         t_mat,
                                         output_sz,
                                         flags=interpolation,
                                         borderMode=cv2.BORDER_CONSTANT)
            else:
                image_t = cv2.warpAffine(image,
                                         t_mat,
                                         output_sz,
                                         flags=interpolation,
                                         borderMode=cv2.BORDER_CONSTANT)

            t_mat_tensor_3x3 = torch.cat(
                (t_mat_tensor.float(), torch.tensor([0., 0., 1.]).view(1, 3)),
                dim=0)
            t_mat_tensor_inverse = t_mat_tensor_3x3.inverse(
            )[:2, :].contiguous()

            sample_pos_inv = torch.mm(sample_grid.view(-1, 3),
                                      t_mat_tensor_inverse.t().float()).view(
                                          *sample_grid.shape[:2], -1)

            if random.random() > .4:
                displacement = get_elastic_transform_shifts(
                    self.alpha, self.sigma, list(sample_grid.shape[:2]))
                shape_1 = torch.as_tensor(
                    [x - 1
                     for x in sample_grid.shape[:2]]).reshape(1, 1, 1, 2) / 2
                base = torch.stack(
                    torch.meshgrid([
                        torch.arange(0, image.shape[0]),
                        torch.arange(0, image.shape[1])
                    ],
                                   indexing='xy'), -1)
                normalized_shifts = (base + displacement.squeeze(0) -
                                     shape_1) / shape_1
                image_t = F.grid_sample(
                    torch.as_tensor(image_t).permute(2, 0, 1).unsqueeze(0),
                    normalized_shifts.clamp_(-1, 1),
                    align_corners=True).squeeze(0).permute(1, 2, 0).numpy()
                sample_pos_inv += displacement.squeeze(0)
                sample_grid[..., :2] -= displacement.squeeze(0)

            sample_pos = torch.mm(sample_grid.view(-1, 3),
                                  t_mat_tensor.t().float()).view(
                                      *sample_grid.shape[:2], -1)

            # Paste object
            if transformation_params.get('border_crop') is not None:
                border_crop = transformation_params.get('border_crop')

                image_t = image_t[border_crop:-border_crop,
                                  border_crop:-border_crop, :]
                sample_pos = sample_pos[border_crop:-border_crop,
                                        border_crop:-border_crop, :]
                sample_pos_inv = sample_pos_inv[border_crop:-border_crop,
                                                border_crop:-border_crop, :]

            sample_pos = sample_pos.numpy()
            sample_pos_inv = sample_pos_inv.numpy()

            # downsample flow
            if not math.isclose(downsample_factor, 1.):
                sample_pos = cv2.resize(
                    sample_pos,
                    None,
                    fx=1.0 / downsample_factor,
                    fy=1.0 / downsample_factor,
                    interpolation=interpolation) / downsample_factor
                sample_pos_inv = cv2.resize(
                    sample_pos_inv,
                    None,
                    fx=1.0 / downsample_factor,
                    fy=1.0 / downsample_factor,
                    interpolation=interpolation) / downsample_factor

            sample_pos = torch.from_numpy(sample_pos).permute(2, 0, 1)
            sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1)

            burst.append(image_t)
            shifts.append(sample_pos)
            shifts_inv.append(sample_pos_inv)

        shifts = torch.stack(shifts)
        shifts_inv = torch.stack(shifts_inv)
        print("len(burst)", len(burst)) # 9
        return burst, image_gt, shifts_inv


def get_elastic_transform_shifts(alpha, sigma, size, small_size=(32, 32)):
    dx = torch.rand([1, 1] + list(small_size)) * 2 - 1
    kx = int(8 * sigma[0] + 1)
    # if kernel size is even we have to make it odd
    if kx % 2 == 0:
        kx += 1
    dx = gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha[0]

    dy = torch.rand([1, 1] + list(small_size)) * 2 - 1
    ky = int(8 * sigma[1] + 1)
    # if kernel size is even we have to make it odd
    if ky % 2 == 0:
        ky += 1
    dy = gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha[1]
    ret = F.interpolate(torch.concat([dx, dy], 1),
                        size=size,
                        mode='bilinear',
                        align_corners=True)
    return ret.permute([0, 2, 3, 1])  # 1 x H x W x 2
