import numpy as np
import tensorflow as tf

class ConvAE(tf.keras.Model):
    def __init__(self, input_shape, encoder, decoder):
        super().__init__(self)
        self.encoder = encoder()
        self.decoder = decoder()
        self.build((1,) + input_shape)

    def call(self, inputs, training=False):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat
