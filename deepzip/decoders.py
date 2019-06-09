
import tensorflow as tf

class EasyDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            64,
            kernel_size=(8, 8),
            strides=4,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.up1 = tf.keras.layers.UpSampling2D((2,2))


        self.conv2 = tf.keras.layers.Conv2DTranspose(
            64,
            kernel_size=(4, 4),
            strides=2,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.up2 = tf.keras.layers.UpSampling2D((2,2))

        self.conv3 = tf.keras.layers.Conv2DTranspose(
            32,
            kernel_size=(3, 3),
            strides=1,
            activation="linear",
            data_format="channels_last",
            padding="same",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.reconstruction = tf.keras.layers.Conv2DTranspose(
            3,
            (3,3),
            activation="linear",
            data_format="channels_last",
            padding="same",
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = self.up1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = self.up2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.reconstruction(x)
        return x
