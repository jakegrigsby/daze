import numpy as np
import tensorflow as tf


class EasyDecoder(tf.keras.models.Model):
    """ Encodes an image using a feedforward neural network.
    """

    def __init__(self, input_shape=(28, 28), latent_dim=32):
        super().__init__()
        num_entries = np.prod(input_shape)

        self.layer1 = tf.keras.layers.Dense(latent_dim, activation="linear")
        self.layer2 = tf.keras.layers.Dense(300, activation="linear")
        self.layer3 = tf.keras.layers.Dense(num_entries, activation="linear")
        self.unflatten = tf.keras.layers.Reshape(input_shape)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = self.layer2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = self.layer3(x)
        x = self.unflatten(x)
        return x
