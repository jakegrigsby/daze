
import tensorflow as tf

class EasyEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(self)
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(8, 8),
            strides=4,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2),
            padding="same",
        )

        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2),
            padding="same",
        )

        self.conv3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=1,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D((2,2),
            padding="same",
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.pool3(x)

        return x
