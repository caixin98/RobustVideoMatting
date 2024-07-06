# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of Adobe
# and its suppliers, if any. The intellectual and technical concepts contained
# herein are proprietary to Adobe and its suppliers and are protected by all
# applicable intellectual property laws, including trade secret and copyright
# laws. Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from Adobe.

import random
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F


class ImageTransforms:

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image: np.array) -> torch.Tensor:
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor:

    def __init__(self):
        pass

    def __call__(self, image: np.array) -> torch.Tensor:
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        if np.max(image) > 1.:
            image /= 255
        if len(image.shape) == 2:
            image = image[..., None]
        if len(image.shape) != 3:
            raise ValueError(
                f'input image has to be of shape [H, W] or [H, W, C]. Got {image.shape} instead'
            )
        if image.shape[-1] not in [1, 3]:
            raise ValueError(
                f'the last dimention of the tensor has to be 1 or 3.')
        image = torch.from_numpy(image).float().permute([2, 0, 1])
        return image


class RandomHorizontalFlip:

    def __init__(self, prob: float = 0.5) -> None:
        self.probability = prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            return image.flip((2, ))
        else:
            return image


def center_crop(frames, crop_sz):
    """
      :param frames: Input frame as tensor
      :param crop_sz: Output crop sz as (rows, cols)
      :return:
      """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()

    shape = frames.shape

    r1 = ((shape[-2] - crop_sz[-2]) // 2).int()
    c1 = ((shape[-1] - crop_sz[-1]) // 2).int()

    r2 = r1 + crop_sz[-2].int().item()
    c2 = c1 + crop_sz[-1].int().item()

    frames_crop = frames[:, r1:r2, c1:c2]

    return frames_crop


def random_resized_crop(frames, crop_sz, scale_range=None, ar_range=None):
    """Randomly take resized crop of input images.
  A crop of size (rows*scale_factor, cols*scale_factor*ar_factor) will be first
  extracted and then scale to (crop_sz, crop_sz). scale_factor and ar_factor
  are uniformly sampled from scale_range and ar_range.
  Args:
      frames: Input frame as [C, H, W] tensor
      crop_sz: Output crop sz as (rows, cols)
      scale_range: A crop of size scale_factor*crop_sz is first extracted
    and resized. The scale_range controls the value of scale_factor. If None,
    no scaling is applied, i.e., directly take random crops of frames.
      ar_range: Aspect ratio range, controling the value of ar_factor. A crop of
    size (rows*scale_factor, cols*scale_factor*ar_factor) will be first
    extracted and then scale to (crop_sz, crop_sz).
  """
    if len(frames.shape) != 3:
        raise ValueError(
            f'input frames has to be in the shape of [C, H, W]. Got {frames.shape} instead.'
        )

    if frames.shape[0] != min(frames.shape):
        raise ValueError(
            f'the frame tensor has to be in [C, H, W] format. Got {frames.shape} instead.'
        )

    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()

    shape = frames.shape

    if ar_range is None:
        ar_factor = 1.0
    else:
        ar_factor = random.uniform(ar_range[0], ar_range[1])

    # Select scale_factor. Ensure the crop fits inside the image
    max_scale_factor = torch.tensor(
        shape[-2:]).float() / (crop_sz * torch.tensor([1.0, ar_factor]))
    max_scale_factor = max_scale_factor.min().item()

    # If frame size is smaller than crop size, just scale it to crop size.
    if max_scale_factor < 1.0:
        scale_factor = max_scale_factor
    elif scale_range is not None:
        scale_factor = random.uniform(scale_range[0],
                                      min(scale_range[1], max_scale_factor))
    else:
        scale_factor = 1.0

    # Extract the crop
    orig_crop_sz = (crop_sz * torch.tensor([1.0, ar_factor]) *
                    scale_factor).floor()

    assert orig_crop_sz[-2] <= shape[-2] and orig_crop_sz[-1] <= shape[
        -1], 'Bug in crop size estimation!'

    r1 = random.randint(0, shape[-2] - orig_crop_sz[-2])
    c1 = random.randint(0, shape[-1] - orig_crop_sz[-1])

    r2 = r1 + orig_crop_sz[0].int().item()
    c2 = c1 + orig_crop_sz[1].int().item()

    frames_crop = frames[:, r1:r2, c1:c2]

    # Resize to crop_sz
    frames_crop_resized = F.interpolate(frames_crop.unsqueeze(0),
                                        size=crop_sz.int().tolist(),
                                        mode='bilinear',
                                        align_corners=True).squeeze(0)
    return frames_crop_resized
