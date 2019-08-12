import pytest
import tensorflow as tf
import numpy as np

from daze.preprocessing import *
import daze as dz


def test_basic_image_normalize():
    img = tf.random.uniform((16, 10, 10, 3), maxval=255, dtype=tf.float32)
    f = basic_image_normalize()
    normalized_img = f(img)
    assert normalized_img.numpy().max() <= 1.0
    assert normalized_img.numpy().min() >= 0.0


def test_random_mask():
    img_shape = (16, 10, 10, 3)
    destruction_coeff = 0.5
    img = tf.random.uniform(img_shape, minval=0.5, maxval=1.0, dtype=tf.float32)
    total_size = tf.size(img).numpy()
    f = random_mask(destruction_coeff, 652)
    masked_img = f(img)
    assert masked_img.shape == img.shape
    num_set_zero = int(total_size * destruction_coeff)
    assert np.count_nonzero(masked_img) == total_size - num_set_zero


def test_gaussian_noise():
    img_shape = (16, 10, 10, 3)
    f = gaussian_noise(mean=0, std=1, seed=852)
    img = tf.random.uniform(img_shape)
    noise_img = f(img)
    assert np.not_equal(img, noise_img).all()

def test_rotation():
    img, _ = dz.data.mnist.load(5)
    top_left_og = img[:, 0, 0, :]
    f = image_rotation(90, 90)
    rot_img = f(img)
    bottom_left_rot = rot_img[:, -1, 0, :]
    # test rotation
    assert top_left_og.all() == bottom_left_rot.all()
    # test original image preserved
    assert top_left_og.all() == img[:, 0, 0, :].all()

def test_horizontal_flip():
    img_shape = (5, 10, 10, 3)
    img = np.random.uniform(size=img_shape)
    f = image_horizontal_flip()
    flipped_img = f(img)
    assert img[:, 0, 0, :].all() == flipped_img[:,0,-1,:].all()

def test_vertical_flip():
    img_shape = (5, 10, 10, 3)
    img = np.random.uniform(size=img_shape)
    f = image_vertical_flip()
    flipped_img = f(img)
    assert img[:, 0, 0, :].all() == flipped_img[:,-1,0,:].all()

