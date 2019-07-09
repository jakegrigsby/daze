"""
Module for custom loss functions
"""

import numpy as np
import tensorflow as tf

from .helpers import *

mse = tf.keras.losses.mean_squared_error

def kl(beta):
    def _beta(**forward_pass_dict):
        logvar = forward_pass_dict['logvar']
        mean = forward_pass_dict['mean']
        return beta*tf.reduce_mean(tf.math.reduce_sum(-.5*(1+logvar - tf.square(mean) - tf.math.exp(logvar))))
    return _beta

def elbo():
    """Evidence Lower Bound"""
    def _elbo(**forward_pass_dict):
        x_hat = forward_pass_dict['x_hat']
        x = forward_pass_dict['x']
        z = forward_pass_dict['z']
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return _elbo

def contractive(coeff):
    # this can't be compiled into a tf.function because of its gradient calculation
    def _contractive(**forward_pass_dict):
        h = forward_pass_dict['h']
        x = forward_pass_dict['x']
        tape = forward_pass_dict['tape']
        dh_dx = tape.gradient(h, x)
        frob_norm = tf.norm(dh_dx)
        return coeff*frob_norm
    return _contractive

def denoising_reconstruction():
    def _denoising(**forward_pass_dict):
        original_x = forward_pass_dict['original_x']
        x_hat = forward_pass_dict['x_hat']
        return mse(x_hat, original_x)
    return _denoising

def reconstruction():
    def _reconstruction(**forward_pass_dict):
        x = forward_pass_dict['x']
        x_hat = forward_pass_dict['x_hat']
        return mse(x, x_hat)
    return _reconstruction

def latent_l1(gamma):
    def _latent_l1(**forward_pass_dict):
        h = forward_pass_dict['h']
        return gamma*tf.norm(h, ord=1)
    return _latent_l1

def sparsity(rho, beta):
    """
    rho is the target sparsity value (~.01), beta is the coefficient for this term.
    """
    def _sparsity(**forward_pass_dict):
        h = forward_pass_dict['h']
        rho_hat = tf.reduce_mean(h, axis=0)
        kl = kl_divergence(rho, rho_hat) 
        return beta*tf.math.reduce_mean(kl)
    return _sparsity
