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

    def compute_gradients(self, x):
        """ Computes gradient of custom loss function.
        """
        with tf.GradientTape() as tape:
          loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))

    def train(self, train_dataset, val_dataset, epochs=10, experiment_name='vae_test'):
        """ Trains the model for a given number of epochs (iterations on a dataset).

        @TODO: implement callbacks, return a History object
        """
        log_dir, checkpoint_dir = self.init_logging(experiment_name)

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in train_dataset:
                print('train_x shape:', train_x.shape)
                gradients, loss = self.compute_gradients(train_x)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
            end_time = time.time()

            if epoch % 1 == 0:
              loss = tf.keras.metrics.Mean()
              for test_x in val_dataset:
                loss(self.compute_loss(test_x))
              total_loss = -loss.result()
              #display.clear_output(wait=False)
              print('Epoch: {}, Test set total loss: {}, '
                    'time elapse for current epoch {}'.format(epoch,
                                                              total_loss,
                                                              end_time - start_time))
        # callbacks = [
            # tf.keras.callbacks.TensorBoard(log_dir, write_graph=True, write_images=True, update_freq='epoch'),
            # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, monitor='val_loss', save_best_only=True, save_weights_only=False, period=1),
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        # ]
