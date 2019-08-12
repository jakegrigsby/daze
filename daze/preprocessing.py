import functools
import math
import random

import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate


def random_mask(destruction_coeff, seed=None):
    """
    Random destruction of data.
    A fraction of the input tensor (determined by destruction_coeff) is
    randomly set to 0.
    """
    if seed:
        np.random.seed(seed)

    def _random_mask(input_batch):
        total_size = tf.size(input_batch).numpy()
        num_set_zero = math.floor(total_size * destruction_coeff)
        mask = np.ones(total_size, dtype=np.float32)
        mask[:num_set_zero] = 0.0
        tf.random.shuffle(mask)
        mask = tf.dtypes.cast(tf.reshape(mask, input_batch.shape), input_batch.dtype)
        return input_batch * mask

    return _random_mask


def gaussian_noise(mean, std, seed=None):
    """
    Inject random gaussian noise with given mean and std.
    """

    def _gaussian_noise(input_batch):
        noise = tf.random.normal(input_batch.shape, mean, std, seed=seed)
        return input_batch + noise

    return _gaussian_noise


def basic_image_normalize():
    def _basic_image_normalize(input_batch):
        return input_batch / 255.0

    return _basic_image_normalize

def image_rotation(min_angle, max_angle, fill_mode='nearest'):
    min_angle = int(min_angle)
    max_angle = int(max_angle)
    assert min_angle <= max_angle
    def _image_rotation(input_batch):
        angle = random.randint(min_angle, max_angle)
        return rotate(input_batch, angle, axes=(-3, -2), mode=fill_mode)
    return _image_rotation

def image_horizontal_flip():
    def _image_horizontal_reflect(input_batch):
        return np.flip(input_batch, axis=-2)
    return _image_horizontal_reflect

def image_vertical_flip():
    def _image_vertical_reflect(input_batch):
        return np.flip(input_batch, axis=-3)
    return _image_vertical_reflect