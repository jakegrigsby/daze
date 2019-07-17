import pytest

import numpy as np

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32


def test_get_batch_encodings_np():
    x, _ = dz.data.cifar10.load(70, "f32")
    x /= 255
    model = dz.Model(Encoder_32x32(latent_dim=2), Decoder_32x32())
    encodings = model.get_batch_encodings(x)
    assert isinstance(encodings, np.ndarray)
    assert encodings.shape[0] == 70
    assert encodings.shape[1] == 2


def test_get_batch_encodings_tf():
    x, _ = dz.data.cifar10.load(70, "f32")
    x /= 255
    x = dz.data.utils.convert_np_to_tf(x, 32)
    model = dz.Model(Encoder_32x32(latent_dim=2), Decoder_32x32())
    encodings = model.get_batch_encodings(x)
    assert isinstance(encodings, np.ndarray)
    assert encodings.shape[0] == 70
    assert encodings.shape[1] == 2


def test_get_batch_encodings_unknown():
    with pytest.raises(ValueError):
        model = dz.Model(Encoder_32x32(latent_dim=2), Decoder_32x32())
        encodings = model.get_batch_encodings([1.0, 2.0, 3.0])
