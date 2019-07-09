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

def test_mnist():
    x_train, x_val = dz.data.mnist.load(64)
    _test_data(x_train, x_val, 'mnist')
    
def test_cifar10():
    x_train, x_val = dz.data.cifar10.load(64)
    _test_data(x_train, x_val, 'cifar10')
"""