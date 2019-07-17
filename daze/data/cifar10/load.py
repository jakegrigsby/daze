import tensorflow as tf
import numpy as np

import daze as dz


def load(size=None, dtype=None):
    """
    Will take a while to run (the first time)
    """
    (x_train, _), (x_val, _) = tf.keras.datasets.cifar10.load_data()
    size = x_train.shape[0] if not size else size
    x_train, x_val = x_train[:size], x_val[:size]
    if dtype:
        dtype = dz.data.utils.parse_dtype(dtype)
        x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    return x_train, x_val
