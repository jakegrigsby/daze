from functools import wraps

import numpy as np
import tensorflow as tf

import deepzip as dz

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


def kl_divergence(a, b):
    return (
        a * tf.math.log(a)
        - a * tf.math.log(b)
        + (1 - a) * tf.math.log(1 - a)
        - (1 - a) * tf.math.log(1 - b)
    )


def sample(model, eps):
    logits = model.decode(eps)
    probs = tf.sigmoid(logits)
    return probs


def sample_random(model):
    eps = tf.random.normal(shape=(100, model.latent_dim))
    return model.sample(eps)
