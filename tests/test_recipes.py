import pytest

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32

x_train, x_val = dz.data.cifar10.load(64)

def test_default():
    model = dz.core.AutoEncoder(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(Encoder_32x32(), Decoder_32x32(), .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_sparse():
    model = dz.recipes.SparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)

def test_contractive():
    model = dz.recipes.ContractiveAutoEncoder(Encoder_32x32(), Decoder_32x32(), .1)
    model.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)