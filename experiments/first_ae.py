import deepzip.decoders
import deepzip.encoders
import deepzip.autoencoders
import deepzip.data.cifar10

model = deepzip.autoencoders.ConvAE(deepzip.encoders.EasyEncoder, deepzip.decoders.EasyDecoder)

x_train, x_val = deepzip.data.cifar10.load()

dz.train.baseline_train(model, x_train, x_val, 10, 'first_ae')


