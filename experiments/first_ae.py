import deepzip as dz

x_train, x_val = dz.data.cifar10.load()

model = dz.autoencoders.ConvAE(input_shape=x_train.shape[-3:], encoder=dz.encoders.EasyEncoder, decoder=dz.decoders.EasyDecoder)
model.summary()

dz.train.baseline_train(model, x_train, x_val, 10, 'first_ae')


