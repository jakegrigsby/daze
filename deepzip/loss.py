"""
Module for custom loss functions
"""

import numpy as np
import tensorflow as tf

@tf.functions
def compute_loss_vae(self, x):
  mean, logvar = self.encode(x)
  z = self.reparameterize(mean, logvar)
  x_logit = self.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def code_frobenius_norm(h, x):
    dh_dx = tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32)
    frob_norm = tf.norm(dh_dx)
    return frob_norm

reconstruction = tf.keras.losses.mean_squared_error

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
