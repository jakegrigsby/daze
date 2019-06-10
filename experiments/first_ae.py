import deepzip as dz

model = dz.autoencoders.ConvAE(dz.encoders.EasyEncoder, dz.decoders.EasyDecoder)

x_train, x_val = dz.data.cifar10.load()

dz.train.baseline_train(model, x_train, x_val, 10, 'first_ae')


