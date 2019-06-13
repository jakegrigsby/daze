import deepzip as dz

x_train, x_val = dz.data.cifar10.load()

model = 'vae'

image_shape = x_train[0].shape
print('x_train shape:' , x_train.shape)
if model == 'convae':
    model = dz.autoencoders.ConvAE(input_shape=image_shape)
elif model == 'vae':
    model = dz.autoencoders.CVAE(input_shape=image_shape)
model.summary()
model.train(x_train, x_val, epochs=10, experiment_name='conv_ae')
