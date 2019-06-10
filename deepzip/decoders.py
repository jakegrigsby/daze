
import tensorflow as tf

class EasyDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(self)
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(7*7*16)
        self.reshape = tf.keras.layers.Reshape(target_shape=(7,7,16))

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            64,
            kernel_size=(2, 2),
            strides=1,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()


        self.conv2 = tf.keras.layers.Conv2DTranspose(
            64,
            kernel_size=(3, 3),
            strides=2,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(
            32,
            kernel_size=(4, 4),
            strides=3,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.reconstruction = tf.keras.layers.Conv2DTranspose(
            3,
            kernel_size=(3,3),
            activation="linear",
            data_format="channels_last",
            padding="same",
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.dense2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.reshape(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = self.reconstruction(x)
        return x
