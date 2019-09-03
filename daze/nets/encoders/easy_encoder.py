import numpy as np
import tensorflow as tf


class EasyEncoder(tf.keras.models.Model):
    """ Encodes an image using a feedforward neural network.
    """

    def __init__(self, input_shape=(28, 28), latent_dim=32):
        super().__init__()
        num_entries = np.prod(input_shape)

        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(
            500, input_dim=num_entries, activation="linear"
        )
        self.layer2 = tf.keras.layers.Dense(300, activation="linear")
        self.layer3 = tf.keras.layers.Dense(latent_dim, activation="linear")

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.layer1(x)
        x = tf.keras.layers.LeakyReLU()(x)
        if training:
            x = tf.keras.layers.Dropout(0.3)(x)
        x = self.layer2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        if training:
            x = tf.keras.layers.Dropout(0.3)(x)
        x = self.layer3(x)
        return x
