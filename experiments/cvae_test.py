import deepzip as dz

x_train, x_val = dz.data.cifar10.load()
model = dz.autoencoders.CVAE(input_shape=x_train.shape[-3:])
model.summary()
dz.train.baseline_train(model, x_train, x_val, 10, 'conv_variational_ae')
