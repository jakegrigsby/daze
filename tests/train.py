import deepzip as dz
from deepzip.autoencoders.encoders import EasyEncoder
from deepzip.autoencoders.decoders import EasyDecoder

import tensorflow as tf

x_train, x_val = dz.data.cifar10.load()

@tf.function
def custom_loss(model, x):
    h = model.encode(x)
    x_hat = model.decode(h)
    return .5 * tf.keras.losses.mean_squared_error(x, x_hat)

model = dz.core.AutoEncoder(EasyEncoder, EasyDecoder)
model.train(x_train, x_val, epochs=10, experiment_name='dae_test')
