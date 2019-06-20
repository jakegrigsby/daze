"""
Module for custom loss functions
"""

from .helpers import log_normal_pdf, reparameterize

import numpy as np
import tensorflow as tf

mse = tf.keras.losses.mean_squared_error

def vae():
    @tf.function
    def _vae(model, x, original_x=None):
        h = model.encode(x)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        z = reparameterize(mean, logvar)
        x_logit = model.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return _vae

def contractive(coeff):
    @tf.function
    def _contractive(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        dh_dx = tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32)
        frob_norm = tf.norm(dh_dx)
        reconstruction = mse(x_hat, x)
        return reconstruction + coeff*frob_norm
    return _contractive

def denoising_contractive(coeff):
    @tf.function
    def _noisy_code_frobenius_norm(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        dh_dx = tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32)
        frob_norm = tf.norm(dh_dx)
        noisy_reconstruction = mse(x_hat, original_x)
        return noisy_reconstruction + coeff*frob_norm
    return _noisy_code_frobenius_norm

def reconstruction_sparsity(coeff):
    @tf.function
    def _reconstruction_sparsity(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        reconstruction = mse(x, x_hat)
        sparsity = tf.norm(h, ord=1)
        return reconstruction + coeff*sparsity
    return _reconstruction_sparsity

def denoising_sparsity(coeff):
    @tf.function
    def _denoising_sparsity(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        noisy_reconstruction = mse(x_hat, original_x)
        sparsity = tf.norm(h, ord=1)
        return noisy_reconstruction + coeff*sparsity
    return _denoising_sparsity

def denoising():
    @tf.function
    def _denoising(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        return mse(x_hat, original_x)
    return _denoising

def reconstruction():
    @tf.function
    def _reconstruction(model, x, original_x=None):
        h = model.encode(x)
        x_hat = model.decode(h)
        return mse(x, x_hat)
    return _reconstruction
    