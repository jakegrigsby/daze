import pytest

import tensorflow as tf

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

def test_mnist_batch():
    x_train, x_val = dz.data.mnist.load(64)
    assert(x_train.shape[0] == 64)
    assert(x_val.shape[0] == 64)
    assert(x_train.shape[1] == 28)
    assert(x_train.shape[1] == x_train.shape[2])

def test_mnist_all():
    x_train, x_val = dz.data.mnist.load()
    assert(x_train.shape[0] > x_val.shape[0])
    
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
    