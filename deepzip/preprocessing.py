import functools

import numpy as np
import tensorflow as tf

def random_mask(destruction_coeff, seed=None):
    """
    Random destruction of data.
    A fraction of the input tensor (determined by destruction_coeff) is
    randomly set to 0.
    """
    def _random_mask(input_batch):
        total_size = tf.size(input_batch).numpy()
        num_set_zero = int(total_size*destruction_coeff)
        idxs = np.random.choice(np.arange(total_size), num_set_zero)
        mask = np.ones(total_size)
        mask[idxs] = 0.
        mask = tf.dtypes.cast(tf.reshape(mask, input_batch.shape), tf.float32)
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
        return input_batch / 255.
    return _basic_image_normalize