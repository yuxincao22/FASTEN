import cv2
from PIL import Image
import random
import torch
import numpy as np
from functools import wraps


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None: return param

    if isinstance(param, (int, float)):
        if low is None:
            param = (-param, param)
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple([bias + x for x in param])

    return tuple(param)

def preserve_shape(func):
    """Preserve shape of the image"""

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function

@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256/255, 1/255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img

def preserve_channel_dim(func):
    """Preserve dummy channel dim."""

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape # H,W,C=(3,1)
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result
    
    return wrapped_function

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Return:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = img.shape[2] if len(img.shape) == 3 else 1 # H,W[,C=(3,>4)]
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index:index+1]
                chunk = process_fn(chunk, **kwargs)
                chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn

@preserve_channel_dim
def shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    H, W = img.shape[:2]
    center = (W/2, H/2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * W
    matrix[1, 2] += dy * H

    warp_affine_fn = _maybe_process_in_chunks(cv2.warpAffine,
                        M=matrix, dsize=(W, H), flags=interpolation, borderMode=border_mode, borderValue=value)
    return warp_affine_fn(img)

def to_jpeg(img, quality=100):
    _, data_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img_decode = cv2.imdecode(data_encode, cv2.IMREAD_COLOR)
    return img_decode


class AntiSpoofingAug(torch.nn.Module):
    def __init__(self, rotation_limit=20, rotation_p=0.25,
                        shift_limit=0.0625, shift_p=0.25,
                        scale_limit=0.15, scale_p=0.25,
                        interpolation=1, interpolation_p=0.25,
                        border_mode=cv2.BORDER_CONSTANT, value=None,
                        gamma_limit=(99, 101), gamma_p=0.25,
                        brightness_value=0.01, brightness_p=0.25,
                        contrast_value=0.01, contrast_p=0.25,
                        flip_horizontal_p=0,
                        jpeg_quality=100, jpeg_p=0) -> None:
        
        super().__init__()
        self.rotation_limit = to_tuple(rotation_limit)
        self.rotation_p = rotation_p

        self.shift_limit = to_tuple(shift_limit)
        self.shift_p = shift_p

        self.scale_limit = to_tuple(scale_limit, bias=1.)
        self.scale_p = scale_p

        self.shift_scale_rotate_p = np.mean((self.rotation_p, self.shift_p, self.scale_p))

        self.interpolation = interpolation
        self.interpolation_p = interpolation_p
        self.border_mode = border_mode
        self.value = value

        self.gamma_limit = to_tuple(gamma_limit)
        self.gamma_p = gamma_p

        self.brightness_value = brightness_value
        self.brightness_p = brightness_p

        self.contrast_value = contrast_value
        self.contrast_p = contrast_p

        self.flip_horizontal_p = flip_horizontal_p

        self.jpeg_quality = jpeg_quality
        self.jpeg_p = jpeg_p

        self.interpolation_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def forward(self, x):
        npx = np.array(x)

        # RandomGamma Aug
        if random.random() < self.gamma_p:
            gamma = random.randint(*self.gamma_limit)
            gamma /= 100.
            npx = gamma_transform(npx, gamma)

        # RandomShiftScaleRotate Aug
        if random.random() < self.shift_scale_rotate_p:
            angle = random.uniform(*self.rotation_limit)
            scale = random.uniform(*self.scale_limit)
            dx = random.uniform(*self.shift_limit)
            dy = random.uniform(*self.shift_limit)
            if random.random() < self.interpolation_p:
                interpolation = random.randint(0, 4)
            else:
                interpolation = self.interpolation

            # Increase the probability
            if random.random() < self.shift_scale_rotate_p:
                scale += 0.3
            
            npx = shift_scale_rotate(npx, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

        # RandomJpeg Aug
        if random.random() < self.jpeg_p:
            npx = to_jpeg(npx, self.jpeg_quality)

        return Image.fromarray(npx.astype('uint8'))


class BiasBoost(torch.nn.Module):
    def __init__(self, shift_limit=0.0125, shift_median=0.5,
                        scale_limit=0.15, scale_median=0.2,
                        positive=True, random=False):
        
        super().__init__()
        self.shift_limit = to_tuple(shift_limit)
        self.shift_median = shift_median

        self.scale_limit = to_tuple(scale_limit, bias=1.)
        self.scale_median = scale_median

        self.positive = positive
        self.random = random

    def forward(self, x):
        npx = np.array(x)

        if self.random:
            direction = random.randint(0, 1)
        elif self.positive:
            direction = 1
        else:
            direction = 0

        scale = self.scale_limit[direction]
        dx = self.shift_limit[direction]
        dy = self.shift_limit[direction]

        npx = shift_scale_rotate(npx, 0, scale, dx, dy, 1, cv2.BORDER_CONSTANT, None)

        return Image.fromarray(npx.astype('uint8'))
