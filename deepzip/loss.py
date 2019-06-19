"""
Module for custom loss functions
"""

import numpy as np 
import tensorflow as tf

@tf.function
def base_vae_loss(model, x_logits, x, z, mean, logvar):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def code_frobenius_norm(model, h, x):
    dh_dx = tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32)
    frob_norm = tf.norm(dh_dx)
    return frob_norm

@tf.function
def reconstruction(model, x):
    h = model.encode(x)
    x_hat = model.decode(h)
    return tf.keras.losses.mean_squared_error(x, x_hat)

