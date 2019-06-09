import numpy as np
import tensorflow as tf

import encoders
import decoders

class ConvAE(tf.keras.Model):
    def __init__(self, input_shape, encoder, decoder):
        super().__init__(self)
        self.encoder = encoder()
        self.decoder = decoder()
        
    def call(self, inputs):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat