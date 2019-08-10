"""
Module for custom loss functions
"""

import numpy as np
import tensorflow as tf
# Currently, we still require the nightly build of tensorflow_probability
import tensorflow_probability as tfp

from .math import *
from .tracing import trace_graph, TRACE_GRAPHS

mse = tf.keras.losses.mean_squared_error


def kl(beta):
    @trace_graph
    def _kl(**forward_pass):
        mean = forward_pass["mean"]
        sigma = forward_pass["sigma"]
        z = forward_pass["z"]
        q_z = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        
        p_z = tfp.distributions.MultivariateNormalDiag(
          loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
          ) # @TODO: would it be faster to pre-store this?
        kl_div = tfp.distributions.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
        print('latent_loss:', latent_loss)
        return beta * latent_loss

    return _kl


def elbo():
    """Evidence Lower Bound"""

    @trace_graph
    def _elbo(**forward_pass):
        x_hat = forward_pass["x_hat"]
        x = forward_pass["x"]
        z = forward_pass["z"]
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    return _elbo


def contractive(coeff):
    # this can't be compiled into a tf.function because of its gradient calculation
    if TRACE_GRAPHS:
        raise ValueError(
            "Autograph tracing not supported for contractive loss. Set dz_trace_graphs environment variable to false:"
            "\t`$ export dz_trace_graphs=False`"
        )

    def _contractive(**forward_pass):
        h = forward_pass["h"]
        x = forward_pass["x"]
        tape = forward_pass["tape_container"].tape
        dh_dx = tape.gradient(h, x)
        frob_norm = tf.norm(dh_dx)
        return coeff * frob_norm

    return _contractive


def denoising_reconstruction():
    @trace_graph
    def _denoising(**forward_pass):
        original_x = forward_pass["original_x"]
        x_hat = forward_pass["x_hat"]
        return mse(x_hat, original_x)

    return _denoising


def reconstruction():
    @trace_graph
    def _reconstruction(**forward_pass):
        x = forward_pass["x"]
        x_hat = forward_pass["x_hat"]
        return mse(x, x_hat)

    return _reconstruction


def latent_l1(gamma):
    @trace_graph
    def _latent_l1(**forward_pass):
        h = forward_pass["h"]
        return gamma * tf.math.reduce_sum(tf.math.abs(h))

    return _latent_l1


def sparsity(rho, beta):
    """
    rho is the target sparsity value (~.01), beta is the coefficient for this term.
    """
    rho = tf.constant(rho)

    @trace_graph
    def _sparsity(**forward_pass):
        h = forward_pass["h"]
        rho_hat = tf.reduce_mean(h, axis=0)
        return beta * tf.keras.losses.kld(rho, rho_hat)

    return _sparsity
