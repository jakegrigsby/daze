import tensorflow as tf
import numpy as np

import daze as dz

def load(size=None, dtype=None, return_labels=False):
    """
    Will take a while to run (the first time)
    """
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    size = x_train.shape[0] if not size else size
    x_train, y_train, x_val, y_val = x_train[:size], y_train[:size], x_val[:size], y_val[:size]
    if dtype:
        dtype = dz.data.utils.parse_dtype(dtype)
        x_train, y_train = x_train.astype(dtype), y_train.astype(dtype)
        x_val, y_val = x_val.astype(dtype), y_val.astype(dtype)
    if return_labels:
        return (x_train, y_train), (x_val, y_val)
    else:
        return x_train, x_val
