import deepzip as dz
from deepzip.autoencoders.encoders import EasyEncoder
from deepzip.autoencoders.decoders import EasyDecoder

import tensorflow as tf

x_train, x_val = dz.data.cifar10.load()

model = dz.recipes.VariationalAutoEncoder(EasyEncoder, EasyDecoder)
print('Model:', model)
model.train(x_train, x_val, epochs=10, experiment_name='vae_test')
