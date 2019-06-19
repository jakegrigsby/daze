import deepzip as dz
from deepzip.autoencoders.encoders import EasyEncoder
from deepzip.autoencoders.decoders import EasyDecoder

x_train, x_val = dz.data.cifar10.load()
image_shape = x_train[0].shape
model = dz.autoencoders.ContractiveAutoEncoder(.25, image_shape, EasyEncoder, EasyDecoder)
model.train(x_train, x_val, epochs=10, experiment_name='dae_test')
