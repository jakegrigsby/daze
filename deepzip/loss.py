"""
Module for custom loss functions
"""

from .helpers import log_normal_pdf, reparameterize

import numpy as np
import tensorflow as tf

mse = tf.keras.losses.mean_squared_error

def vae():
    @tf.function
    def _vae(tape, original_x, x, h, mean, logvar, z, x_logit):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return _vae

def contractive(coeff):
    # this can't be compiled into a tf.function because of its gradient calculation
    def _contractive(tape, original_x, x, h, x_hat):
        dh_dx = tape.gradient(h, x)
        frob_norm = tf.norm(dh_dx)
        return coeff*frob_norm
    return _contractive

def denoising():
    @tf.function
    def _denoising(tape, original_x, x, h, x_hat):
        return mse(x_hat, original_x)
    return _denoising

def reconstruction():
    @tf.function
    def _reconstruction(tape, original_x, x, h, x_hat):
        return mse(x, x_hat)
    return _reconstruction

def sparsity(coeff):
    @tf.function
    def _sparsity(tape, original_x, x, h, x_hat):
        return coeff*tf.norm(h, ord=1)
    return _sparsity