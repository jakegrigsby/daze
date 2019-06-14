import os
import time

import numpy as np
import tensorflow as tf

import deepzip as dz

class VariationalAutoEncoder(BaselineAE):
    def call(self, inputs, training=False):
        mean, logvar = self.encode(inputs)
        h = self.reparameterize(mean, logvar)
        x_hat = self.decode(h)
        return x_hat

    @tf.function
    def sample(self, eps=None):
        """ Allows users to sample from the latent space. """
        if eps is None:
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits

    @tf.function
    def compute_loss(self, x):
      mean, logvar = self.encode(x)
      z = self.reparameterize(mean, logvar)
      x_logits = self.decode(z)
      loss = dz.loss.base_vae_loss(x_logits, x, z, mean, logvar)
      return loss
