"""
Module for custom loss functions
"""

from .helpers import log_normal_pdf, reparameterize

import numpy as np
import tensorflow as tf

@tf.function
def compute_loss_vae(model, x, original_x=None):
    h = model.encode(x)
    mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
    z = reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def code_frobenius_norm(model, x, original_x=None):
    dh_dx = tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32)
    frob_norm = tf.norm(dh_dx)
    return frob_norm

@tf.function
def noisy_reconstruction(model, x, original_x=None):
    h = model.encode(x)
    x_hat = model.decode(h)
    return tf.keras.losses.mean_squared_error(x_hat, original_x)

@tf.function
def reconstruction(model, x, original_x=None):
    h = model.encode(x)
    x_hat = model.decode(h)
    return tf.keras.losses.mean_squared_error(x, x_hat)
