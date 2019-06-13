import os
import time

import numpy as np
import tensorflow as tf

from . import BaselineAE

from deepzip.autoencoders import encoders, decoders

class CVAE(BaselineAE):
    """ A convolutional variational autoencoder. """

    def __init__(self, input_shape):
        super().__init__(self)

        self.encoder = encoders.EasyEncoder()
        self.decoder = decoders.EasyDecoder()
        self.network_trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        self.build((1,) + input_shape)

    def call(self, inputs, training=False):
        mean, logvar = self.encode(inputs)
        h = self.reparameterize(mean, logvar)
        x_hat = self.decode(h)
        return x_hat

    def sample(self, eps=None):
        """ Allows users to sample from the latent space. """
        if eps is None:
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs

        return logits

    def compute_loss(self, x):
      mean, logvar = self.encode(x)
      z = self.reparameterize(mean, logvar)
      x_logit = self.decode(z)

      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
      logpz = log_normal_pdf(z, 0., 0.)
      logqz_x = log_normal_pdf(z, mean, logvar)
      return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)
