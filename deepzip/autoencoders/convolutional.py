import numpy as np
import tensorflow as tf

from deepzip.autoencoders import encoders, decoders

class ConvAE(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__(self)
        self.encoder = encoders.EasyEncoder()
        self.decoder = decoders.EasyDecoder()
        self.build((1,) + input_shape)

    def call(self, inputs, training=False):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat
