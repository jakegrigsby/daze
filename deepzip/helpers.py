""" Helper functions for internal things. Not exposed as part of
    the public API. """

import numpy as np
import tensorflow as tf

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

def softmax(x):
    return 1.*x / tf.math.reduce_sum(x)

def kl_divergence(a, b):
    return a * tf.math.log(a) - a * tf.math.log(b) + (1 - a) * tf.math.log(1 - a) - (1 - a) * tf.math.log(1 - b)