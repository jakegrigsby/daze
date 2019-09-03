from functools import wraps

import numpy as np
import tensorflow as tf

import daze as dz

###################
# Misc Math Funcs #
###################


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

def softmax(x):
    return 1.0 * x / tf.math.reduce_sum(x)

def random_normal(shape):
    """Convenience wrapper to prevent tf imports on gan demo"""
    if not isinstance(shape, list):
        shape = list(shape)
    return tf.random.normal(shape)
