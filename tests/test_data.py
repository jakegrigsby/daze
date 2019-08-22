import pytest
import tensorflow as tf
import numpy as np

import daze as dz
from daze.nets.encoders import EasyEncoder
from daze.nets.decoders import EasyDecoder

###########
## MNIST ##
###########

def test_mnist_batch():
    x_train, x_val = dz.data.mnist.load(64)
    assert x_train.shape[0] == 64
    assert x_val.shape[0] == 64
    assert x_train.shape[1] == 28
    assert x_train.shape[1] == x_train.shape[2]

def test_mnist_all():
    x_train, x_val = dz.data.mnist.load()
    assert x_train.shape[0] > x_val.shape[0]

def test_mnist_return_labels():
    (x_train, y_train), (x_val, y_val) = dz.data.mnist.load(return_labels=True)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

def test_mnist_float():
    x_train, x_val = dz.data.mnist.load(64, dtype=np.float32)
    assert x_train.dtype == np.float32
    x_train, x_val = dz.data.mnist.load(64, dtype="f32")
    assert x_train.dtype == np.float32

#############
## CIFAR10 ##
#############

def test_cifar10_batch():
    x_train, x_val = dz.data.cifar10.load(64)
    assert x_train.shape[0] == 64
    assert x_val.shape[0] == 64
    assert x_train.shape[1] == 32
    assert x_train.shape[1] == x_train.shape[2]

def test_cifar10_all():
    x_train, x_val = dz.data.cifar10.load()
    assert x_train.shape[0] > x_val.shape[0]

def test_cifar10_return_labels():
    (x_train, y_train), (x_val, y_val) = dz.data.cifar10.load(return_labels=True, size=10)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

def test_cifar10_float():
    x_train, x_val = dz.data.cifar10.load(dtype="f32", size=10)
    assert x_train.dtype == np.float32
    x_train, x_val = dz.data.cifar10.load(dtype=np.float32, size=10)
    assert x_train.dtype == np.float32

#############
## CIFAR100 ##
#############

def test_cifar100_batch():
    x_train, x_val = dz.data.cifar100.load(64)
    assert x_train.shape[0] == 64
    assert x_val.shape[0] == 64
    assert x_train.shape[1] == 32
    assert x_train.shape[1] == x_train.shape[2]

def test_cifar100_all():
    x_train, x_val = dz.data.cifar100.load()
    assert x_train.shape[0] > x_val.shape[0]

def test_cifar100_return_labels():
    (x_train, y_train), (x_val, y_val) = dz.data.cifar100.load(return_labels=True, size=10)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

def test_cifar100_float():
    x_train, x_val = dz.data.cifar100.load(dtype="f32", size=10)
    assert x_train.dtype == np.float32
    x_train, x_val = dz.data.cifar100.load(dtype=np.float32, size=10)
    assert x_train.dtype == np.float32


####################
## BOSTON HOUSING ##
####################

def test_boston_batch():
    x_train, x_val = dz.data.boston_housing.load(64)
    assert x_train.shape[0] == 64
    assert x_val.shape[0] == 64
    assert x_train.shape[1] == 13

def test_boston_float32():
    x_train, x_val = dz.data.boston_housing.load(dtype="f32", size=10)
    assert x_train.dtype == np.float32
    x_train, x_val = dz.data.boston_housing.load(dtype=np.float32, size=10)
    assert x_train.dtype == np.float32

def test_boston_all():
    x_train, x_val = dz.data.boston_housing.load(dtype="f32")
    assert x_train.shape[0] > x_val.shape[0]

def test_boston_return_labels():
    (x_train, y_train), (x_val, y_val) = dz.data.boston_housing.load(return_labels=True, size=10)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]


###################
## FASHION MNIST ##
###################

def test_fashionmnist_all():
    x_train, x_val = dz.data.fashionmnist.load(dtype="f32")
    assert x_train.shape[1:] == (28, 28, 1)
    assert x_train.shape[0] > x_val.shape[0]

def test_fashionmnist_batch():
    x_train, x_val = dz.data.fashionmnist.load(64)
    assert x_train.shape[0] == 64
    assert x_val.shape[0] == 64

def test_fashionmnist_float():
    x_train, x_val = dz.data.fashionmnist.load(dtype=np.float32, size=10)
    assert x_train.dtype == np.float32
    x_train, x_val = dz.data.fashionmnist.load(dtype="f32", size=10)
    assert x_train.dtype == np.float32

def test_fashionmnist_return_labels():
    (x_train, y_train), (x_val, y_val) = dz.data.fashionmnist.load(return_labels=True, size=10)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

##############
## DSPRITES ##
##############

def test_dsprites_all():
    x = dz.data.dsprites.load()
    assert x.shape[0] > 7000
    assert x.shape[1:] == (64, 64)
    assert x.dtype == np.uint8

def test_dsprites_batch():
    x = dz.data.dsprites.load(size=8)
    assert x.shape[1:] == (64, 64)
    assert x.shape[0] == 8

def test_dsprites_float():
    x = dz.data.dsprites.load(size=10, dtype=np.float32)
    assert x.dtype == np.float32
    x = dz.data.dsprites.load(size=10, dtype="f32")
    assert x.dtype == np.float32

def test_dsprites_return_labels():
    x, y = dz.data.dsprites.load(size=10, return_labels=True)
    assert y.dtype == np.float64
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 6

############
## CHAIRS ##
############

def test_chairs_all():
    x = dz.data.chairs.load()
    assert x

###########
## ESC50 ##
###########

def test_esc50_all():
    x = dz.data.esc50.load()
    assert x.shape[0] == 2000
    assert x.shape[1] == 220500

def test_esc50_batch():
    x = dz.data.esc50.load(size=10)
    assert x.shape[0] == 10
    assert x.shape[1] == 220500

def test_esc50_float():
    x = dz.data.esc50.load(dtype="f32")
    assert x.dtype == np.float32


##########
## MISC ##
##########

def test_np_to_tf():
    x_train, _ = dz.data.cifar10.load(64)
    x_train, batch_count = dz.data.utils.convert_np_to_tf(
        x_train, batch_size=32, return_batch_count=True
    )
    assert isinstance(x_train, tf.data.Dataset)
    assert batch_count == 2

def test_default_dtypes():
    """
    Check defaul datatypes of provided datasets.
    """
    mnist, _ = dz.data.mnist.load(1)
    assert mnist.dtype == np.uint8
    cifar, _ = dz.data.cifar10.load(1)
    assert cifar.dtype == np.uint8
    boston, _ = dz.data.boston_housing.load(1)
    assert boston.dtype == np.float64
    fashion, _ = dz.data.fashionmnist.load(1)
    assert fashion.dtype == np.int64


def test_parse_dtype():
    f = dz.data.utils.parse_dtype
    assert f("f") == np.float32
    assert f("f16") == np.float16
    assert f("f32") == np.float32
    assert f("f64") == np.float64
    assert f("float") == np.float32
    assert f("fl16") == np.float16
    with pytest.raises(ValueError):
        f("f100000000")

    assert f("i") == np.int32
    assert f("i16") == np.int16
    assert f("i32") == np.int32
    assert f("i64") == np.int64
    assert f("int") == np.int32
    assert f("in16") == np.int16
    with pytest.raises(ValueError):
        f("i100000000")

    assert f("u") == np.uint32
    assert f("u8") == np.uint8
    assert f("u16") == np.uint16
    assert f("u32") == np.uint32
    assert f("u64") == np.uint64
    assert f("uint") == np.uint32
    assert f("ui16") == np.uint16
    with pytest.raises(ValueError):
        f("u100000000")

    assert f(np.float16) == np.float16

    with pytest.raises(ValueError):
        f("23")


def test_get_byte_count():
    f = dz.data.utils.get_byte_count
    assert not f("f")
    assert not f("i")
    assert not f("u")
    assert f("f8") == 8
    assert f("f16") == 16
    assert f("fljlhlh32") == 32
