import math
import os
import time

import deepzip as dz
import tensorflow as tf

from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

import matplotlib.pyplot as plt

show_plots = False

x_train, x_val = dz.data.mnist.load(1024)

latent_dim = 22

folder_name = "vae-test-{}".format(int(time.time()))
os.mkdir(folder_name)
print("Created directory {}.".format(folder_name))
num_epochs = 10
vae = dz.recipes.VariationalAutoEncoder(
    EasyEncoder(latent_dim=latent_dim), 
    EasyDecoder()
)

num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, int(latent_dim/2)]
)

# for matplotlib
num_rows = math.floor(math.sqrt(num_examples_to_generate))
num_cols = math.ceil(num_examples_to_generate / num_rows)


epoch = 0


def sample_from_model(model, eps):
    logits = model.decode(eps)
    probs = tf.sigmoid(logits)
    return probs


def generate_and_save_images(model):
    global epoch
    epoch += 1
    predictions = sample_from_model(model, random_vector_for_generation)
    
    fig = plt.figure(figsize=(4, 4))
    print('predictions[0].shape:', predictions[0].shape)
    for i in range(predictions.shape[0]):
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig("{}/image_at_epoch_{:04d}.png".format(folder_name, epoch))

    if show_plots:
        plt.show()


_i = 0


def callback(model, **kwargs):
    generate_and_save_images(model)


vae.train(
    x_train,
    x_val,
    epochs=num_epochs,
    verbosity=2,
    callbacks=[callback],
)
