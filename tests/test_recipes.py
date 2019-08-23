import pytest
import copy

import tensorflow as tf
import numpy as np

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder
from daze.callbacks import *
from daze.preprocessing import basic_image_normalize

x_train_np, x_val_np = dz.data.cifar10.load(64, "f32")
sample_img = x_train_np[0]
x_train_np /= 255
x_val_np /= 255

x_train = dz.data.utils.convert_np_to_tf(x_train_np, batch_size=32)
x_val = dz.data.utils.convert_np_to_tf(x_val_np, batch_size=32)

#######################
# Test Training Loops #
#######################

def train(model, callbacks):
    model.train(
        x_train,
        x_val,
        callbacks=callbacks,
        save_path="tests/saves",
        epochs=2,
        verbosity=0,
    )

def make_callbacks(model):
    callbacks = [checkpoints(1), 
                tensorboard_loss_scalars(), 
                tensorboard_gradient_histograms(2),
                tensorboard_image_reconstruction(x_train_np[:3,...]),
                tensorboard_latent_space_plot(x_train_np[:10,...]),
                tensorboard_trace_graph(model.encode, np.expand_dims(sample_img, 0)),
                ]
    return callbacks


def test_default():
    model = dz.AutoEncoder(ConvolutionalEncoder(3), CifarDecoder())
    cbs = make_callbacks(model)
    train(model, cbs)

def test_gan():
    model = dz.GAN(CifarDecoder(), 100, ConvolutionalEncoder(1))
    train(model, None)

def test_vae():
    model = dz.recipes.VariationalAutoEncoder(ConvolutionalEncoder(), CifarDecoder())
    cbs = make_callbacks(model)
    train(model, cbs)

def test_denoising():
    model = dz.recipes.DenoisingAutoEncoder(ConvolutionalEncoder(), CifarDecoder(), gamma=0.1)
    cbs = make_callbacks(model)
    train(model, cbs)

def test_klsparse():
    model = dz.recipes.KlSparseAutoEncoder(
        ConvolutionalEncoder(), CifarDecoder(), rho=0.01, beta=0.1
    )
    cbs = make_callbacks(model)
    train(model, cbs)

def test_l1sparse():
    model = dz.recipes.L1SparseAutoEncoder(ConvolutionalEncoder(), CifarDecoder(), gamma=0.1)
    cbs = make_callbacks(model)
    train(model, cbs)

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
        cbs = make_callbacks(model)
        train(model, cbs)

def test_beta_vae():
    model = dz.recipes.BetaVariationalAutoEncoder(
        ConvolutionalEncoder(), CifarDecoder(), beta=1.1
    )
    cbs = make_callbacks(model)
    train(model, cbs)
