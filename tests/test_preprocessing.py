
import pytest
import tensorflow as tf
import numpy as np

from deepzip.preprocessing import *

def test_basic_image_normalize():
    img = tf.random.uniform((10,10,3), maxval=255, dtype=tf.float32)
    f = basic_image_normalize()
    normalized_img = f(img)
    assert(normalized_img.numpy().max() <= 1.)
    assert(normalized_img.numpy().min() >= 0.)

def test_random_mask():
    img_shape = (10, 10, 3)
    destruction_coeff = .5
    img = tf.random.uniform(img_shape, minval=.5, maxval=1., dtype=tf.float32)
    total_size = tf.size(img).numpy()
    f = random_mask(destruction_coeff, 652)
    masked_img = f(img)
    assert(masked_img.shape == img.shape)
    num_set_zero = int(total_size*destruction_coeff)
    assert(np.count_nonzero(masked_img) == total_size-num_set_zero)

def test_gaussian_noise():
    img_shape = (10,10,3)
    f = gaussian_noise(mean=0,std=1,seed=852)
    img = tf.random.uniform(img_shape)
    noise_img = f(img)
    assert(np.not_equal(img, noise_img).all())

