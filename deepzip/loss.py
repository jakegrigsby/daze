"""
Module for custom loss functions
"""

import numpy as np
import tensorflow as tf

from .helpers import *

mse = tf.keras.losses.mean_squared_error


def kl(beta):
    def _beta(**kwargs):
        logvar = kwargs['logvar']
        mean = kwargs['mean']
        return beta*tf.reduce_mean(tf.math.reduce_sum(-.5*(1+logvar - tf.square(mean) - tf.math.exp(logvar))))
    return _beta

def contractive(coeff):
    # this can't be compiled into a tf.function because of its gradient calculation
    def _contractive(**kwargs):
        h = kwargs['h']
        x = kwargs['x']
        tape = kwargs['tape']
        dh_dx = tape.gradient(h, x)
        frob_norm = tf.norm(dh_dx)
        return coeff*frob_norm
    return _contractive

def denoising_reconstruction():
    def _denoising(**kwargs):
        original_x = kwargs['original_x']
        x_hat = kwargs['x_hat']
        return mse(x_hat, original_x)
    return _denoising

def reconstruction():
    def _reconstruction(**kwargs):
        x = kwargs['x']
        x_hat = kwargs['x_hat']
        return mse(x, x_hat)
    return _reconstruction

def latent_l1(beta):
    def _latent_l1(**kwargs):
        h = kwargs['h']
        return beta*tf.norm(h, ord=1)
    return _latent_l1

def sparsity(rho, beta):
    """
    rho is the target sparsity value (~.01), beta is the coefficient for this term.
    """
    def _sparsity(**kwargs):
        h = kwargs['h']
        rho_hat = tf.reduce_mean(h, axis=0)
        kl = kl_divergence(rho, rho_hat) 
        return beta*tf.math.reduce_mean(kl)
    return _sparsity
