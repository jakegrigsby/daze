import math

import deepzip as dz
import tensorflow as tf

from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

import matplotlib.pyplot as plt

show_plots = False

x_train, x_val = dz.data.cifar10.load(64)
latent_dim = 16

vae = dz.recipes.VariationalAutoEncoder(EasyEncoder(), EasyDecoder(), latent_dim=latent_dim)

num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

print('random_vector_for_generation:', random_vector_for_generation)

epoch = 0
def generate_and_save_images(model):
  global epoch
  epoch += 1
  predictions = model.sample(random_vector_for_generation)
  # print('predictions:', predictions)
  fig = plt.figure(figsize=(4,4))
  
  num_rows = math.floor(math.sqrt(num_examples_to_generate))
  num_cols = math.ceil(num_examples_to_generate / num_rows)

  for i in range(predictions.shape[0]):
      plt.subplot(num_rows, num_cols, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  
  if show_plots:
    plt.show()
 

_i = 0
def callback(): 
  generate_and_save_images(vae, random_vector_for_generation)

vae.train(x_train, x_val, epochs=10, experiment_name='test_default', verbosity=1, callbacks=[generate_and_save_images])
    