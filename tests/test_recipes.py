import pytest

import tensorflow as tf

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder
from daze.callbacks import checkpoints, tensorboard
from daze.preprocessing import basic_image_normalize

x_train, x_val = dz.data.cifar10.load(64, "f32")
x_train /= 255
x_train = dz.data.utils.convert_np_to_tf(x_train, batch_size=32)
x_val /= 255
x_val = dz.data.utils.convert_np_to_tf(x_val, batch_size=32)
callbacks = [checkpoints(1), tensorboard()]

#######################
# Test Training Loops #
#######################


def test_default():
    model = dz.Model(ConvolutionalEncoder(), CifarDecoder())
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )


def test_vae():
    model = dz.recipes.VariationalAutoEncoder(ConvolutionalEncoder(), CifarDecoder())
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )


def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(ConvolutionalEncoder(), CifarDecoder(), gamma=0.1)
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )


def test_klsparse():
    model = dz.recipes.KlSparseAutoEncoder(
        ConvolutionalEncoder(), CifarDecoder(), rho=0.01, beta=0.1
    )
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )


def test_l1sparse():
    model = dz.recipes.L1SparseAutoEncoder(ConvolutionalEncoder(), CifarDecoder(), gamma=0.1)
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )


def test_contractive():
    if dz.tracing.TRACE_GRAPHS:
        with pytest.raises(ValueError):
            model = dz.recipes.ContractiveAutoEncoder(
                ConvolutionalEncoder(), CifarDecoder(), gamma=0.1
            )
    else:
        model = dz.recipes.ContractiveAutoEncoder(
            ConvolutionalEncoder(), CifarDecoder(), gamma=0.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="tests/saves",
            epochs=1,
            verbosity=0,
        )


def test_beta_vae():
    model = dz.recipes.BetaVariationalAutoEncoder(
        ConvolutionalEncoder(), CifarDecoder(), beta=1.1
    )
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=1,
        verbosity=0,
    )
