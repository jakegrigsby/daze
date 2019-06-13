import abc
import os
import time

import numpy as np
import tensorflow as tf

from deepzip.autoencoders import encoders, decoders

class BaselineAE(tf.keras.Model):
    """ A basic autoencoder.
    """

    def call(self, x):
        """ Approximates x by encoding and decoding it.
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, h):
        return

    def init_logging(self, experiment_name):
        """ Sets up log directories for training.
        """
        directory = 'data/' + experiment_name + str(int(time.time()))

        # Setup checkpoints and logging
        checkpoint_dir = os.path.join(directory, 'checkpoints')
        os.makedirs(checkpoint_dir)

        log_dir = os.path.join(directory, 'logs')
        os.makedirs(log_dir)

        return log_dir, checkpoint_dir

    def compute_loss(self, x):
        """ Computes loss during training. Default loss function is MSE.
        """
        h = self.encode(x)
        x_hat = self.decode(z)
        return tf.losses.mean_squared_error(x, x_hat)

    @tf.function
    def compute_gradients(self, x):
        """ Computes gradient of custom loss function.
        """
        with tf.GradientTape() as tape:
          loss = self.compute_loss(x)
        return tape.gradient(loss, self.network_trainable_variables), loss

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))
    
    def create_dataset(self, numpy_dataset):
        dataset_size = numpy_dataset.shape[0]
        dataset = tf.cast(numpy_dataset/255, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(dataset_size + 1).batch(32)
        return dataset

    def train(self, train_dataset, val_dataset, epochs=10, experiment_name='vae_test'):
        """ Trains the model for a given number of epochs (iterations on a dataset).

        @TODO: implement callbacks, return a History object
        """
        log_dir, checkpoint_dir = self.init_logging(experiment_name)

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam()

        train_dataset = self.create_dataset(train_dataset)
       
        for epoch in range(epochs):
            start_time = time.time()
            for (batch, (images)) in enumerate(train_dataset):
                gradients, loss = self.compute_gradients(images)
                self.apply_gradients(optimizer, gradients, self.network_trainable_variables)
            end_time = time.time()
        
        val_dataset = self.create_dataset(val_dataset)
        val_loss = tf.keras.metrics.Mean()
        for images in val_dataset:
            val_loss(self.compute_loss(images))
        val_loss = -loss.result()
        print('Epoch: {}, Test set total loss: {}, '
                'time elapse for current epoch {}'.format(epoch,
                                                            val_loss,
                                                            end_time - start_time))
