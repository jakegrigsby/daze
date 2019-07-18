import pytest

import tensorflow as tf

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder


def test_get_batch_encodings_np():
    x, _ = dz.data.cifar10.load(70, "f32")
    x /= 255
    model = dz.Model(ConvolutionalEncoder(latent_dim=2), CifarDecoder())
    encodings = model.get_batch_encodings(x)
    assert isinstance(encodings, tf.Tensor)
    assert encodings.numpy().shape[0] == 70
    assert encodings.numpy().shape[1] == 2


def test_get_batch_encodings_tf():
    x, _ = dz.data.cifar10.load(70, "f32")
    x /= 255
    x = dz.data.utils.convert_np_to_tf(x, 32)
    model = dz.Model(ConvolutionalEncoder(latent_dim=2), CifarDecoder())
    encodings = model.get_batch_encodings(x)
    assert isinstance(encodings, tf.Tensor)
    assert encodings.numpy().shape[0] == 70
    assert encodings.numpy().shape[1] == 2


def test_get_batch_encodings_unknown():
    with pytest.raises(ValueError):
        model = dz.Model(ConvolutionalEncoder(latent_dim=2), CifarDecoder())
        encodings = model.get_batch_encodings([1.0, 2.0, 3.0])
