import tensorflow as tf
import numpy as np

import daze as dz


def load(size=None, dtype=None):
    (x_train, _), (x_val, _) = tf.keras.datasets.mnist.load_data()
    size = x_train.shape[0] if not size else size
    x_train, x_val = x_train[:size], x_val[:size]
    if dtype:
        dtype = dz.data.utils.parse_dtype(dtype)
        x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    return x_train, x_val
