
import pytest
import tensorflow as tf
import numpy as np

import deepzip as dz
from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

"""
def _test_data(x_train, x_val, data_name):
    input_shape = x_train[1:].shape
    encoder = EasyEncoder(input_shape=input_shape)
    decoder = EasyDecoder(input_shape=input_shape)
    
    model = dz.core.Model(encoder, decoder)
    model.train(x_train, x_val, epochs=1, experiment_name='test_{}'.format(data_name), verbosity=0)
"""

def test_default_dtypes():
    """
    Check defaul datatypes of provided datasets.
    """
    mnist, _ = dz.data.mnist.load(1)
    assert(mnist.dtype == np.uint8)
    cifar, _ = dz.data.cifar10.load(1)
    assert(cifar.dtype == np.uint8)
    boston, _ = dz.data.boston_housing.load(1)
    assert(boston.dtype == np.float64)

def test_parse_dtype():
    f = dz.data.utils.parse_dtype
    assert(f('f') == np.float32)
    assert(f('f16') == np.float16)
    assert(f('f32') == np.float32)
    assert(f('f64') == np.float64)
    assert(f('float') == np.float32)
    assert(f('fl16') == np.float16)

    assert(f('i') == np.int32)
    assert(f('i16') == np.int16)
    assert(f('i32') == np.int32)
    assert(f('i64') == np.int64)
    assert(f('int') == np.int32)
    assert(f('in16') == np.int16)

    assert(f('u') == np.uint32)
    assert(f('u8') == np.uint8)
    assert(f('u16') == np.uint16)
    assert(f('u32') == np.uint32)
    assert(f('u64') == np.uint64)
    assert(f('uint') == np.uint32)
    assert(f('ui16') == np.uint16)

    assert(f(np.float16) == np.float16)

def test_get_byte_count():
    f = dz.data.utils.get_byte_count
    assert(not f('f'))
    assert(not f('i'))
    assert(not f('u'))
    assert(f('f8') == 8)
    assert(f('f16') == 16)
    assert(f('fljlhlh32') == 32)

def test_mnist_batch():
    x_train, x_val = dz.data.mnist.load(64)
    assert(x_train.shape[0] == 64)
    assert(x_val.shape[0] == 64)
    assert(x_train.shape[1] == 28)
    assert(x_train.shape[1] == x_train.shape[2])

def test_mnist_all():
    x_train, x_val = dz.data.mnist.load()
    assert(x_train.shape[0] > x_val.shape[0])

def test_mnist_float():
    x_train, x_val = dz.data.mnist.load(64, dtype=np.float32)
    
def test_cifar10_batch():
    x_train, x_val = dz.data.cifar10.load(64)
    assert(x_train.shape[0] == 64)
    assert(x_val.shape[0] == 64)
    assert(x_train.shape[1] == 32)
    assert(x_train.shape[1] == x_train.shape[2])

def test_cifar10_all():
    x_train, x_val = dz.data.cifar10.load()
    assert(x_train.shape[0] > x_val.shape[0])

def test_boston_batch():
    x_train, x_val = dz.data.boston_housing.load(64)
    assert(x_train.shape[0] == 64)
    assert(x_val.shape[0] == 64)
    assert(x_train.shape[1] == 13)

def test_boston_all():
    x_train, x_val = dz.data.boston_housing.load()
    assert(x_train.shape[0] > x_val.shape[0])
    