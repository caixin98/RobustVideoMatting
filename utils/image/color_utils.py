# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of Adobe
# and its suppliers, if any. The intellectual and technical concepts contained
# herein are proprietary to Adobe and its suppliers and are protected by all
# applicable intellectual property laws, including trade secret and copyright
# laws. Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from Adobe.
"""Color manipulation functions."""

import abc
import enum
from typing import Optional, Union

import gin
import numpy as np
import torch as th

# yapf: disable



# Convert to PCS from Linear ProPhoto.
# https://git.corp.adobe.com/acr/acr/blob/master/camera_raw/dng_sdk/source/dng_color_space.cpp#L760
PCS_FROM_PROPHOTO = np.array(
    [[0.7977, 0.1352, 0.0313],
     [0.288 , 0.7119, 0.0001],
     [0.    , 0.    , 0.8249]],
    dtype=np.float32)

# Convert to PCS from Linear sRGB.
# https://git.corp.adobe.com/acr/acr/blob/master/camera_raw/dng_sdk/source/dng_color_space.cpp#L254
PCS_FROM_sRGB = np.array(
    [[0.4361, 0.3851, 0.1431],
     [0.2225, 0.7169, 0.0606],
     [0.0139, 0.0971, 0.7141]],
    dtype=np.float32)

# Convert to PCS from Linear Display P3.
# https://git.corp.adobe.com/acr/acr/blob/56cee5504eaf685d7d955042b2e34564aebce08f/camera_raw/dng_sdk/source/dng_color_space.cpp#L757
PCS_FROM_DISPLAYP3 = np.array(
    [[0.5151, 0.2920, 0.1571],
     [0.2412, 0.6922, 0.0666],
     [-0.0010, 0.0419, 0.7843]],
    dtype=np.float32)

# yapf: enable


class GammaFunction(abc.ABC):
    """Base class for gamma functions."""

    @staticmethod
    @abc.abstractmethod
    def forward(x: th.FloatTensor) -> th.FloatTensor:
        """Applies the forward gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the forward gamma
        function.
    """
        pass

    @staticmethod
    @abc.abstractmethod
    def inverse(x: th.FloatTensor) -> th.FloatTensor:
        """Applies the inverse gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the inverse gamma
        function.
    """
        pass


@gin.register(module='tone_curve')
class SimpleGammaFunction(GammaFunction):
    """Simple gamma function class.

  This class implements a simple gamma function and its inverse.
  """

    GAMMA = 1 / 2.2

    @classmethod
    def forward(cls, x: th.FloatTensor) -> th.FloatTensor:
        sign = th.sign(x)
        y = th.abs(x)
        # Approximation to have valid gradients when x = 0.
        y = th.pow(y + 1e-8, cls.GAMMA) - 1e-8**cls.GAMMA
        return sign * y

    @classmethod
    def inverse(cls, x: th.FloatTensor) -> th.FloatTensor:
        sign = th.sign(x)
        y = th.abs(x)
        # Approximation to have valid gradients when x = 0.
        y = th.pow(y + 1e-8, (1.0 / cls.GAMMA)) - 1e-8**(1.0 / cls.GAMMA)
        return sign * y


@gin.register(module='tone_curve')
class SRGBGammaFunction(GammaFunction):
    """sRGB gamma function class.

  This class implements the sRGB gamma function and its inverse.
  """

    @staticmethod
    def forward(x: th.FloatTensor) -> th.FloatTensor:
        y = th.where(
            x <= 0.0031308, x * 12.92,
            1.055 * th.pow(th.clamp(x, min=0.0031308), 1.0 / 2.4) - 0.055)
        y = th.where(x > 1, (x - 1.0) * 1.055 / 2.4 + 1.0, y)
        return y

    @staticmethod
    def inverse(x: th.FloatTensor) -> th.FloatTensor:
        y = th.where(
            x <= 0.0031308 * 12.92, x * (1.0 / 12.92),
            th.pow(
                (th.clamp(x, min=0.0031308 * 12.92) + 0.055) * (1.0 / 1.055),
                2.4))
        y = th.where(x > 1, (x - 1.0) * 2.4 / 1.055 + 1.0, y)
        return y


@gin.register(module='tone_curve')
class PQGammaFunction(GammaFunction):
    """PQ gamma function class.

  This class implements the normalized PQ gamma function and its inverse based
  on the SMPTE ST 2084 standard. Maps normalized linear light in [0, 1] to
  [0, 1] through a non-linear encoding.

  Note that this function assumes the input is a float32 linear image, where 1.0
  represents SDR white, which is 203 nits (used by the forthcoming ISO 22028-5
  standard for representing HDR still photos).

  Note that before applying the PQ function, the input should be normalized to
  range [0, 1], where 1 represents 10000 nits. For an overrange input where 1.0
  represents SDR white, SDR white corrsponds to 203 nits, i.e., the input should
  be normalized by multiplying 203 / 10000 first.

  It's implemented to be *almost* gradient-safe by applying the abs function to
  the input tensor and preserving the sign in the end. However, the gradient is
  still undefined when the input is 0, and there is no easy way to fix this as
  the curve is highly non-linear when close to 0.

  This function is probably not useful for training anyway. The gradient of
  the PQ function is bad as the curve is highly non-linear.

  TODO(zxia): think about how to make this differentiable at 0.

  The forward PQ function is given by:
  L = (C1 + C2 * (Y ^ M1)) / (1 + C3 * (Y ^ M1)) ^ M2

  The inverse PQ function is given by:
  Y = ((L ^ (1 / M2)) - C1) / (C2 - C3 * (L ^ (1 / M2))) ^ (1 / M1)

  Constants:
  C1 = 0.8359375 (107 / 128)
  C2 = 18.8515625 (2413 / 128)
  C3 = 18.6875 (2392 / 128)
  M1 = 0.1593017578125 (1305 / 8192)
  M2 = 78.84375 (2523 / 32)

  NORM_FACTOR = 10000.0 / 203.0

  https://git.corp.adobe.com/acr/acr/blob/f6a56ce698ca4f5cf4fcc14cf921406f483cca70/camera_raw/cr_sdk/source/cr_color_space.h#L589
  """

    C1 = 0.8359375  # 107 / 128
    C2 = 18.8515625  # 2413 / 128
    C3 = 18.6875  # 2392 / 128
    M1 = 0.1593017578125  # 1305 / 8192
    M2 = 78.84375  # 2523 / 32

    NORM_FACTOR = 10000.0 / 203.0

    @classmethod
    def forward(cls, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the forward PQ gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the forward gamma
        function.
    """
        sign = th.sign(x)
        x = th.abs(x)
        x = x / cls.NORM_FACTOR
        y = th.pow(x, cls.M1)
        result = th.pow((cls.C1 + cls.C2 * y) / (1.0 + cls.C3 * y), cls.M2)
        return sign * result

    @classmethod
    def inverse(cls, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the inverse PQ gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the inverse gamma
        function.
    """
        sign = th.sign(x)
        x = th.abs(x)
        y = th.pow(x, 1.0 / cls.M2)
        result = th.pow(
            th.clamp(y - cls.C1, min=0.0) / (cls.C2 - cls.C3 * y),
            1.0 / cls.M1)
        result = result * cls.NORM_FACTOR
        return sign * result


def _apply_ccm_to_img(img: th.FloatTensor, ccm: np.ndarray) -> th.FloatTensor:
    ccm = th.from_numpy(ccm.T).to(img.device)
    # Raise ccm to 1x3x3x1x1 and img to Bx3x1xHxW for broadcasting.
    img = th.sum(ccm[None, :, :, None, None] * img[:, :, None, :, :], axis=1)
    return img


@gin.register(module='tone_curve')
class _SRGBLogGammaFunction(GammaFunction):
    """sRGB Log gamma function.

  that maps linear [0, 16] to non-linear [0, 1]. Specifically, [0, 1] is mapped
  to [0, split_scale] using a standard sRGB gamma curvr and [1, 16] is
  mapped to [split_scale, 1] using an affine log. The curve consists of
  three parts:

  1. Linear sRGB: For x <= 0.0031308, the function behaves linearly, following
     the standard sRGB curve.
  2. Non-linear sRGB: For 0.0031308 < x <= 1.0, the function follows a
     non-linear sRGB curve, as defined in the sRGB standard.
  3. Logarithmic encoding: For x > 1.0, the function encodes values in a
     logarithmic space, with an optional slope extension for values x > 16.0.

  The curve is controlled by the parameters a, b, and c, and the
  scaling factor `split_scale`. For values over 16, if `slope_ext` is set to
  true, they are mapped using a linear curve with slope same as the slope of
  the log-curve at x=16. If false, they are mapped using the same log-curve.

  The implementation is gradient-safe, by applying the abs function to
  the input tensor and preserving the sign in the end.

  The forward method applies the encoding, and the inverse method applies the
  decoding, converting the encoded values back to the original space.

  Do not use this class directly, use the actual gamma functions created from
  this class instead.

  https://git.corp.adobe.com/acr/acr/blob/f6a56ce698ca4f5cf4fcc14cf921406f483cca70/camera_raw/cr_sdk/source/cr_tone_utils.h#L1650-L1651
  """

    def __init__(self, a: float, b: float, c: float, split_scale: float,
                 slope_ext: bool):
        self.a = a
        self.b = b
        self.c = c
        self.split_scale = split_scale
        self.slope_ext = slope_ext

        if abs(self.a * np.log(16.0 + self.b) + self.c - 1.0) > 1e-4:
            raise ValueError(
                "Invalid parameters for sRGB Log gamma function. It "
                "must map 16 to 1.")

    def forward(self, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the forward Encode_sRGB_Log gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the forward gamma
        function.
    """
        sign = th.sign(x)
        x = th.abs(x)

        # First part is sRGB
        result = th.where(
            x <= 0.0031308, x * 12.92 * self.split_scale,
            self.split_scale *
            (1.055 * th.pow(th.clamp(x, min=0.0031308), 1.0 / 2.4) - 0.055))

        # Process (1, 16] with affine log
        if not self.slope_ext:
            result = th.where(x > 1.0,
                              self.a * th.log(x + self.b) + self.c, result)
        else:
            result = th.where((x > 1.0) & (x <= 16.0),
                              self.a * th.log(x + self.b) + self.c, result)

            # Slope extension
            scale = self.a / (16.0 + self.b)
            offset = (1.0 - scale * 16.0)
            result = th.where(x > 16.0, scale * x + offset, result)

        return result * sign

    def inverse(self, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the inverse Encode_sRGB_Log gamma function to the input tensor.

    Args:
      x (th.FloatTensor): The input tensor to be processed.

    Returns:
      th.FloatTensor: The output tensor after applying the inverse gamma
        function.
    """
        sign = th.sign(x)
        x = th.abs(x)

        scale = 12.92 * self.split_scale
        result = th.where(
            x <= 0.0031308 * scale, x / scale,
            th.pow(
                (th.clamp(x / self.split_scale, min=0.0031308 * scale) + 0.055)
                / 1.055, 2.4))

        if not self.slope_ext:
            result = th.where(x > self.split_scale,
                              th.exp((x - self.c) / self.a) - self.b, result)
        else:
            result = th.where((x > self.split_scale) & (x <= 1.0),
                              th.exp((x - self.c) / self.a) - self.b, result)

            # Slope extension
            scale = self.a / (16.0 + self.b)
            offset = (1.0 - scale * 16.0)
            result = th.where(x > (16.0 * scale + offset),
                              (x - offset) / scale, result)

        return result * sign


# Hybrid sRGB-Log encoding that maps [0, 1] to [1, 0.51] with a sGRB curve and
# [1, 16] to [0.51, 1] with a log-curve.
SRGBLog51GammaFunction = _SRGBLogGammaFunction(0.15781, -0.296081, 0.565406,
                                               0.51, True)

# Hybrid sRGB-Log encoding that maps [0, 1] to [1, 0.6375] with a sGRB curve and
# [1, 16] to [0.6375, 1] with a log-curve.
SRGBLog6375GammaFunction = _SRGBLogGammaFunction(0.0951221, -0.660562,
                                                 0.740276, 0.6375, True)


class Division4EVGammaFunction(GammaFunction):
    """
  Fast curve-based tone map gamma function that maps [0, 16] to [0, 1].

  Using a division-based method. Values above 16 are smoothly mapped using
  the same function. This gamma function is designed to handle the tone mapping
  of images with a fast curve-based approach. The method is implemented to be
  gradient-safe.
  """

    A = 1.0 / (16.0 * 16.0)
    A_TOP = 4.0 * A
    A_BOTTOM = 1.0 / (2.0 * A)

    @classmethod
    def forward(cls, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the forward gamma function to the input tensor."""
        sign = th.sign(x)
        x = th.abs(x)
        result = x * (1.0 + cls.A * x) / (1.0 + x)
        return result * sign

    @classmethod
    def inverse(cls, x: th.FloatTensor) -> th.FloatTensor:
        """Applies the inverse gamma function to the input tensor."""
        sign = th.sign(x)
        x = th.abs(x)
        xm1 = x - 1.0
        result = cls.A_BOTTOM * (xm1 + th.sqrt(xm1 * xm1 + cls.A_TOP * x))
        return result * sign


class _ColorSpace:
    """Class for representing a color space.

  Do not use this class directly, use the actual ColorSpace created from this
  class instead.
  """

    def __init__(
        self,
        name: str,
        to_pcs_ccm: np.ndarray,
        gamma_func: Optional[GammaFunction] = None,
    ):
        """Initializes the color space object.

    Args:
      name (str): The name of the color space.
      to_pcs_ccm (np.ndarray): The matrix to convert to the PCS color space.
      gamma_func (GammaFunction, optional): The gamma function to use for the
        color space. Defaults to None.
    """
        self.name = name
        self.to_pcs_ccm = to_pcs_ccm
        self.from_pcs_matrix = np.linalg.inv(to_pcs_ccm)
        self.gamma_func = gamma_func

    def from_pcs(self, x: th.FloatTensor) -> th.FloatTensor:
        """Converts an image from the PCS color space to this color space.

    Args:
      x (th.FloatTensor): The image to convert, shape: [B, C, H, W].

    Returns:
      th.FloatTensor: Output image in this color space.
    """
        x = _apply_ccm_to_img(x, self.from_pcs_matrix)
        if self.gamma_func is not None:
            x = self.gamma_func.forward(x)
        return x

    def to_pcs(self, x: th.FloatTensor) -> th.FloatTensor:
        """Converts a color from this color space to the PCS color space.

    Args:
      x (th.FloatTensor): The image to convert, shape: [B, C, H, W].

    Returns:
      th.FloatTensor: Output image in PCS color space.
    """
        if self.gamma_func is not None:
            x = self.gamma_func.inverse(x)
        x = _apply_ccm_to_img(x, self.to_pcs_ccm)
        return x

    def to_pcs_luminance(self, x: th.FloatTensor) -> th.FloatTensor:
        """Converts an image from this color space to luminance (CIE Y).

    Args:
      x (th.FloatTensor): The image to convert, shape: [B, C, H, W].

    Returns:
      th.FloatTensor: Output image in luminance.
    """
        # Broadcast gains of shape (3,) to (1, 3, 1, 1) to match (B, C, H, W).
        gains = th.Tensor(self.to_pcs_ccm[1, :][None, :, None,
                                                None]).to(x.device)
        return th.sum(gains * x, axis=1, keepdim=True)


class _LAB(_ColorSpace):
    """Class for representing the CIE L*a*b* color space."""

    # XYZ coordinates of D50 white point.
    D50_XYZ = np.array([0.96430, 1.0, 0.82510])
    DELTA = 6.0 / 29.0
    M = 7.787037  # 1 / 3 * (29 / 6)**2

    def __init__(self):
        """Initializes the CIE L*a*b* color space object."""
        super().__init__("LAB", np.diag(self.D50_XYZ), None)

    def from_pcs(self, x: th.FloatTensor) -> th.FloatTensor:
        """Converts an image from the PCS color space to this color space.

    Args:
      x (th.FloatTensor): The image to convert, shape: [B, C, H, W].

    Returns:
      th.FloatTensor: Output image in this color space.
    """
        x = _apply_ccm_to_img(x, ccm=self.from_pcs_matrix)

        # Note that we do not clamp here, to preseve the gradient.

        x = th.where(x > self.DELTA**3, x**(1.0 / 3.0),
                     self.M * x + 16.0 / 116.0)

        x = [
            116.0 * x[:, 1, :, :] - 16.0,
            500.0 * (x[:, 0, :, :] - x[:, 1, :, :]),
            200.0 * (x[:, 1, :, :] - x[:, 2, :, :])
        ]
        x = th.stack(x, axis=1)
        return x

    def to_pcs(self, x: th.FloatTensor) -> th.FloatTensor:
        """Converts a color from this color space to the PCS color space.

    Args:
      x (th.FloatTensor): The image to convert, shape: [B, C, H, W].

    Returns:
      th.FloatTensor: Output image in PCS color space.
    """
        x = th.stack([(x[:, 0, :, :] + 16) / 116.0 + x[:, 1, :, :] / 500.0,
                      (x[:, 0, :, :] + 16) / 116.0,
                      (x[:, 0, :, :] + 16) / 116.0 - x[:, 2, :, :] / 200.0],
                     axis=1)
        x = th.where(x > self.DELTA, x**3, (x - 16.0 / 116.0) / self.M)
        x = _apply_ccm_to_img(x, ccm=self.to_pcs_ccm)
        return x


class ColorSpace(enum.Enum):
    """Color space enum.

  The actual ColorSpace class can be accessed with ColorSpace.<name> or
  ColorSpace["<name>"].
  """

    # pylint: disable=invalid-name

    Linear_sRGB = _ColorSpace("Linear sRGB", PCS_FROM_sRGB, None)

    Linear_ProPhoto = _ColorSpace("Linear ProPhoto", PCS_FROM_PROPHOTO, None)

    Linear_DisplayP3 = _ColorSpace("Linear Display P3", PCS_FROM_DISPLAYP3,
                                   None)

    Gamma_sRGB = _ColorSpace("sRGB", PCS_FROM_sRGB, SRGBGammaFunction)

    # Gamma Display P3 shares the same transfer function as sRGB.
    Gamma_DisplayP3 = _ColorSpace("Display P3", PCS_FROM_DISPLAYP3,
                                  SRGBGammaFunction)

    # HDR friendly sRGB color space that uses Perceptual Quantizer (PQ) as
    # the transfer function.
    PQ_sRGB = _ColorSpace("PQ sRGB", PCS_FROM_sRGB, PQGammaFunction)

    # HDR friendly Display P3 color space that uses Perceptual Quantizer (PQ) as
    # the transfer function.
    PQ_DisplayP3 = _ColorSpace("PQ Display P3", PCS_FROM_DISPLAYP3,
                               PQGammaFunction)

    # CIE LAB color space.
    LAB = _LAB()

    # pylint: enable=invalid-name

    def __new__(cls, color_space):
        obj = object.__new__(cls)
        obj._value = color_space
        return obj

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __call__(self, *args, **kwargs):
        return self._value(*args, **kwargs)


def convert(x: th.FloatTensor, from_color_space: Union[str, ColorSpace],
            to_color_space: Union[str, ColorSpace]):
    """Converts an image from one color space to another.

  Args:
    x (th.FloatTensor): The image to convert, shape: [B, C, H, W].
    from_color_space (str, ColorSpace): The color space to convert from.
    to_color_space (str, ColorSpace): The color space to convert to.

  Returns:
    th.FloatTensor: Output image in the target color space.
  """
    from_color_space = ColorSpace[from_color_space] if isinstance(
        from_color_space, str) else from_color_space
    to_color_space = ColorSpace[to_color_space] if isinstance(
        to_color_space, str) else to_color_space
    x = from_color_space.to_pcs(x)
    x = to_color_space.from_pcs(x)
    return x


def put_image(img: th.FloatTensor,
              to_color_space: Union[str, ColorSpace] = ColorSpace.Gamma_sRGB,
              clamp: bool = False) -> th.FloatTensor:
    """Converts the image from Linear ProPhoto to a chosen color space."""
    # Clamp negative values but preserves overrange values.
    img = th.clamp(img, min=0.)
    img = convert(img, ColorSpace.Linear_ProPhoto, to_color_space)
    if clamp:
        img = th.clamp(img, 0., 1.)
    return img


def get_image(img: th.FloatTensor,
              from_color_space: Union[str, ColorSpace] = ColorSpace.Gamma_sRGB,
              clamp: bool = False) -> th.FloatTensor:
    """Converts the color space of an image from gamma sRGB to Linear ProPhoto."""
    # Clamp negative values but preserves overrange values.
    img = th.clamp(img, min=0.)
    img = convert(img, from_color_space, ColorSpace.Linear_ProPhoto)
    if clamp:
        img = th.clamp(img, 0., 1.)
    return img


# some how gin enum constant is broken
@gin.configurable()
def Division4EV(func=Division4EVGammaFunction()):
    return func.forward


@gin.configurable()
def SRGBLog51(func=SRGBLog51GammaFunction):
    return func.forward


@gin.configurable()
def SRGBLog6375(func=SRGBLog6375GammaFunction):
    return func.forward
