import deepzip as dz
import tensorflow as tf

from deepzip.nets.encoders import EasyEncoder
from deepzip.nets.decoders import EasyDecoder

x_train, x_val = dz.data.cifar10.load(64)
latent_dim = 256

vae = dz.recipes.VariationalAutoEncoder(EasyEncoder(), EasyDecoder(), latent_dim=latent_dim)

num_examples_to_generate = 10
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

def generate_and_save_images(model, epoch, test_input):
  print('sample:', model.sample)
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
 

for epoch in range(10):
    vae.train(x_train, x_val, epochs=1, experiment_name='test_default', verbosity=0)
    generate_and_save_images(vae, epoch, random_vector_for_generation)