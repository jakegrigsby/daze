import pytest

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

x_train, x_val = dz.data.cifar10.load(64)

encoder = EasyEncoder()
decoder = EasyDecoder()

def test_default():
    model = dz.core.AutoEncoder(encoder, decoder)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(encoder, decoder, .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_sparse():
    model = dz.recipes.SparseAutoEncoder(encoder,decoder, .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_contractive():
    model = dz.recipes.ContractiveAutoEncoder(encoder, decoder, .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)