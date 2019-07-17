import tensorflow as tf
import numpy as np

import deepzip as dz


def load(size=None, dtype=None):
    (x_train, _), (x_val, _) = tf.keras.datasets.mnist.load_data()
    size = x_train.shape[0] if not size else size
    x_train, x_val = x_train[:size], x_val[:size]
    
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_val = tf.cast(x_val, tf.float32) / 255.0
    
    return x_train, x_val
