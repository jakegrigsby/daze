import os
import time
import functools

import numpy as np
import tensorflow as tf

from .loss import reconstruction
from . import helpers

class AutoEncoder(tf.keras.Model):
    """ A basic autoencoder.
    """
    def __init__(self, encode_block, decode_block, preprocessing_steps=[], call_func='ae', loss_funcs=[reconstruction()]):
        super().__init__()
        self.encoder, self.decoder = encode_block(), decode_block()
        self.preprocessing_steps = preprocessing_steps
        self.loss_funcs = loss_funcs
        self.choose_call_func(call_func)

    def choose_call_func(self, call_func):
        if call_func == 'ae':
            self.call = self.call_ae
        elif call_func == 'vae':
            self.call = self.call_vae
        else:
            raise ValueError(f"Unrecognized value for call_func: {call_func}. Options are 'ae' and 'vae'.")

    def preprocess_input(self, x):
        for func in self.preprocessing_steps:
            x = func(x)
        return x
    
    def compute_loss(self, *args):
        loss = 0
        for loss_func in self.loss_funcs:
            loss += loss_func(*args)
        return loss
    
    @property
    def trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables
    
    def call_ae(self, original_x, x):
        """ Approximates x by encoding and decoding it.
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return original_x, x, h, x_hat
    
    def call_vae(self, original_x, x):
        h = self.encode(x)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        z = helpers.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        return original_x, x, h, mean, logvar, z, x_logit

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def init_logging(self, experiment_name):
        """ Sets up log directories for training.
        """
        directory = os.path.join('data/', experiment_name)
        # get unique number for this run
        i = 0
        while os.path.exists(directory + f"_{i}"):
            i += 1
        directory += f"_{i}"
        # Setup checkpoints and logging
        checkpoint_dir = os.path.join(directory, 'checkpoints')
        os.makedirs(checkpoint_dir)
        log_dir = os.path.join(directory, 'logs')
        os.makedirs(log_dir)
        return log_dir, checkpoint_dir

    def compute_gradients(self, x, original_x):
        """ Computes gradient of custom loss function.
        """
        with tf.GradientTape() as tape:
            call_result = self.call(original_x, x)
            loss = self.compute_loss(*call_result)
        grad = tape.gradient(loss, self.trainable_variables)
        return grad, loss

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))
    
    def create_dataset(self, numpy_dataset):
        dataset_size = numpy_dataset.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(numpy_dataset)
        dataset = dataset.shuffle(dataset_size + 1).batch(64)
        return dataset
    
    def train(self, train_dataset, val_dataset, epochs=10, experiment_name='vae_test', verbosity=1):
        """ Trains the model for a given number of epochs (iterations on a dataset).

        @TODO: implement callbacks, return a History object
        """
        log_dir, checkpoint_dir = self.init_logging(experiment_name)

        optimizer = tf.keras.optimizers.Adam()

        train_dataset = self.create_dataset(train_dataset)
        val_dataset = self.create_dataset(val_dataset)
       
        for epoch in range(epochs):
            start_time = time.time()
            for (batch, (original_x)) in enumerate(train_dataset):
                processed_x = self.preprocess_input(original_x)
                gradients, loss = self.compute_gradients(processed_x, original_x)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
            end_time = time.time()
        
            val_loss = tf.keras.metrics.Mean()
            for original_x in val_dataset:
                processed_x = self.preprocess_input(original_x)
                val_loss.update_state(self.compute_loss(*self.call(original_x, processed_x)))
            val_loss = val_loss.result().numpy()
            if verbosity == 1:
                print('Epoch: {}, Test set total loss: {}, '
                        'time elapse for current epoch {}'.format(epoch,
                                                                    val_loss,
                                                                    end_time - start_time))
