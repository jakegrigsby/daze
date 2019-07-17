import tensorflow as tf


class Encoder_32x32(tf.keras.models.Model):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(4, 4),
            strides=4,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=2,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(2, 2),
            strides=1,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="linear")
        self.dense2 = tf.keras.layers.Dense(latent_dim, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.dense2(x)

        return x


EncoderCifar10 = Encoder_32x32
