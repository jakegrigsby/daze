import deepzip as dz
from deepzip.autoencoders.encoders import EasyEncoder
from deepzip.autoencoders.decoders import EasyDecoder

import tensorflow as tf

x_train, x_val = dz.data.cifar10.load()

@tf.function
def custom_loss(model, x):
    h = model.encode(x)
    x_hat = model.decode(h)
    return .5 * dz.loss.reconstruction(x, x_hat)

model = dz.core.AutoEncoder(EasyEncoder, EasyDecoder, loss=custom_loss)
model.train(x_train, x_val, epochs=10, experiment_name='dae_test')
