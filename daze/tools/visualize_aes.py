import pytest
import time

import deepzip as dz
import matplotlib.pyplot as plt
import tensorflow as tf

from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

x_train, x_val = dz.data.cifar10.load()
num_epochs = 20
models = []

print("AutoEncoder (Vanilla)")
model = dz.core.AutoEncoder(EasyEncoder(), EasyDecoder())
model.train(x_train, x_val, epochs=num_epochs, experiment_name="vis_test", verbosity=1)
models.append(model)

# print('VariationalAutoEncoder')
# model = dz.recipes.VariationalAutoEncoder(EasyEncoder(), EasyDecoder())
# model.train(x_train, x_val, epochs=num_epochs, experiment_name='vis_test', verbosity=0)
# models.append(model)

print("DenoisingAutoEncoder")
model = dz.recipes.DenoisingAutoEncoder(EasyEncoder(), EasyDecoder(), 0.1)
model.train(x_train, x_val, epochs=num_epochs, experiment_name="vis_test", verbosity=1)
models.append(model)

print("SparseAutoEncoder")
model = dz.recipes.SparseAutoEncoder(EasyEncoder(), EasyDecoder(), 0.1)
model.train(x_train, x_val, epochs=num_epochs, experiment_name="vis_test", verbosity=1)
models.append(model)

# print('ContractiveAutoEncoder')
# model = dz.recipes.ContractiveAutoEncoder(EasyEncoder(), EasyDecoder(), .1)
# model.train(x_train, x_val, epochs=num_epochs, experiment_name='vis_test', verbosity=0)
# models.append(model)


x = tf.expand_dims(x_val[0], axis=0)
print("x shape:", x.shape)

num_models = len(models)
num_rows = num_models
num_cols = 2

fig = plt.figure(figsize=(4, 4))
for i in range(num_models):
    auto_encoder = models[i]
    print("trying model:", type(auto_encoder).__name__)
    x_hat = auto_encoder.call(x)
    print("x_hat.shape:", x_hat.shape)

    x_image = x[0]
    x_hat_image = x_hat[0]  # * 255

    plt.subplot(num_rows, num_cols, i * 2 + 1)
    plt.imshow(x_image)
    plt.axis("off")

    plt.subplot(num_rows, num_cols, i * 2 + 2)
    plt.imshow(x_hat_image)
    plt.axis("off")

filename = "vis-{}-{}.png".format(num_epochs, int(time.time()))
plt.savefig(filename)
plt.show()
