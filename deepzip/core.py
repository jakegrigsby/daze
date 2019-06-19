import os
import time
import functools

import numpy as np
import tensorflow as tf

from .loss import reconstruction

class AutoEncoder(tf.keras.Model):
    """ A basic autoencoder.
    """
    def __init__(self, encode_block, decode_block, preprocessing_steps=None, loss=reconstruction):
        super().__init__()
        self.encoder, self.decoder = encode_block(), decode_block()
        self.preprocess_input = preprocessing_steps
        self.compute_loss = functools.partial(loss, self)

    def preprocess_input(self, x):
        for func in self.preprocessing_steps:
            x = func(x)
        return x
    
    @property
    def trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def call(self, x):
        """ Approximates x by encoding and decoding it.
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

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

    def compute_gradients(self, x):
        """ Computes gradient of custom loss function.
        """
        with tf.GradientTape() as tape:
          loss = self.compute_loss(x)
        grad = tape.gradient(loss, self.trainable_variables)
        return grad, loss

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))
    
    def create_dataset(self, numpy_dataset):
        dataset_size = numpy_dataset.shape[0]
        dataset = tf.cast(numpy_dataset/255, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(dataset_size + 1).batch(64)
        return dataset
    
    def process_input(self, inputs):
        """Apply any preprocessing to a batch of input data (noise, augmentation)"""
        return inputs

    def train(self, train_dataset, val_dataset, epochs=10, experiment_name='vae_test'):
        """ Trains the model for a given number of epochs (iterations on a dataset).

        @TODO: implement callbacks, return a History object
        """
        log_dir, checkpoint_dir = self.init_logging(experiment_name)

        optimizer = tf.keras.optimizers.Adam()

        train_dataset = self.create_dataset(train_dataset)
        val_dataset = self.create_dataset(val_dataset)
       
        for epoch in range(epochs):
            start_time = time.time()
            for (batch, (images)) in enumerate(train_dataset):
                images = self.process_input(images)
                gradients, loss = self.compute_gradients(images)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
            end_time = time.time()
        
            val_loss = tf.keras.metrics.Mean()
            for images in val_dataset:
                val_loss.update_state(self.compute_loss(images))
            val_loss = val_loss.result().numpy()
            print('Epoch: {}, Test set total loss: {}, '
                    'time elapse for current epoch {}'.format(epoch,
                                                                val_loss,
                                                                end_time - start_time))
