import os
os.chdir('.')
import pytest

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32

x_train, x_val = dz.data.cifar10.load(64)

def test_default():
    model = dz.core.Model(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(Encoder_32x32(), Decoder_32x32(), .1)
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_klsparse():
    model = dz.recipes.KlSparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), rho=.01, beta=.1)
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_l1sparse():
    model = dz.recipes.L1SparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=.1)
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_contractive():
    model = dz.recipes.ContractiveAutoEncoder(Encoder_32x32(), Decoder_32x32(), .1)
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)

def test_beta_vae():
    model = dz.recipes.BetaVariationalAutoEncoder(Encoder_32x32(), Decoder_32x32(), beta=1.1)
    model.train(x_train, x_val, save_path='tests/saves', epochs=1, verbosity=0)
