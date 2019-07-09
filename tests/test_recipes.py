import pytest

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32
from deepzip.callbacks import checkpoints, tensorboard
from deepzip.preprocessing import basic_image_normalize

x_train, x_val = dz.data.cifar10.load(64, 'f32')
callbacks = [checkpoints(1), tensorboard()]

def test_default():
    model = dz.core.Model(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()])
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()])
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()], gamma=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_klsparse():
    model = dz.recipes.KlSparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()], rho=.01, beta=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_l1sparse():
    model = dz.recipes.L1SparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()], gamma=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_contractive():
    model = dz.recipes.ContractiveAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()], gamma=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_beta_vae():
    model = dz.recipes.BetaVariationalAutoEncoder(Encoder_32x32(), Decoder_32x32(), preprocessing_steps=[basic_image_normalize()], beta=1.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)
