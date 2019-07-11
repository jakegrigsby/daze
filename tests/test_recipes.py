import pytest

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32
from deepzip.callbacks import checkpoints, tensorboard
from deepzip.preprocessing import basic_image_normalize

x_train, x_val = dz.data.cifar10.load(64, 'f32')
x_train /= 255
x_train = dz.data.utils.np_convert_to_tf(x_train, batch_size=32)
x_val /= 255
x_val = dz.data.utils.np_convert_to_tf(x_val, batch_size=32)
callbacks = [checkpoints(1), tensorboard()]

#######################
# Test Training Loops #
#######################

def test_default():
    model = dz.Model(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(Encoder_32x32(), Decoder_32x32())
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_klsparse():
    model = dz.recipes.KlSparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), rho=.01, beta=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_l1sparse():
    model = dz.recipes.L1SparseAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_contractive():
    if dz.tracing.TRACE_GRAPHS:
        with pytest.raises(ValueError):
            model = dz.recipes.ContractiveAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=.1)
    else:
        model = dz.recipes.ContractiveAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=.1)
        model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)

def test_beta_vae():
    model = dz.recipes.BetaVariationalAutoEncoder(Encoder_32x32(), Decoder_32x32(), beta=1.1)
    model.train(x_train, x_val, callbacks=callbacks, save_path='tests/saves', epochs=1, verbosity=0)