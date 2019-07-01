import os
import time
import functools
import math

import numpy as np
import tensorflow as tf

from .loss import reconstruction
from . import helpers

class AutoEncoder(tf.keras.Model):
    """ A basic autoencoder.
    """
    def __init__(self, encoder, decoder, preprocessing_steps=[], call_func='ae', loss_funcs=[reconstruction()]):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
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
    
    def compute_loss(self, original_x, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            forward_pass = self.call(original_x, x)
            loss = 0
            for loss_func in self.loss_funcs:
                loss += loss_func(tape, *forward_pass)
        return loss, tape
    
    def predict(self, x):
        return self.call(x, x)[-1]
    
    @property
    def weights(self):
        return [self.encoder.get_weights(), self.decoder.get_weights()]
    
    @weights.setter
    def weights(self, new_weights):
        self.encoder.set_weights(new_weights[0])
        self.decoder.set_weights(new_weights[1])
    
    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.encoder.save_weights(os.path.join(path, "encoder"))
        self.decoder.save_weights(os.path.join(path, "decoder"))
    
    def load_weights(self, path):
        self.encoder.load_weights(os.path.join(path, "encoder"))
        self.decoder.load_weights(os.path.join(path, "decoder"))
 
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
        directory = os.path.join('saves/', experiment_name)
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

    def compute_gradients(self, original_x, x):
        """ Computes gradient of custom loss function.
        """
        loss, tape = self.compute_loss(original_x, x)
        grad = tape.gradient(loss, self.trainable_variables)
        return grad, loss

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))
    
    def create_dataset(self, numpy_dataset, batch_size):
        dataset_size = numpy_dataset.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(numpy_dataset)
        dataset = dataset.shuffle(dataset_size + 1).batch(batch_size)
        return dataset
    
    def train(self, train_dataset, val_dataset, epochs=10, experiment_name='test', verbosity=1, callbacks=None, batch_size=64):
        """ Trains the model for a given number of epochs (iterations on a dataset).

        @TODO: implement callbacks, return a History object
        """
        log_dir, checkpoint_dir = self.init_logging(experiment_name)

        optimizer = tf.keras.optimizers.Adam()

        train_set_size = train_dataset.shape[0]
        batch_count = math.ceil(train_set_size / batch_size)
        train_dataset = self.create_dataset(train_dataset, batch_size)
        val_dataset = self.create_dataset(val_dataset, batch_size)

        log_writer = tf.summary.create_file_writer(log_dir)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)       
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        for epoch in range(epochs):
            start_time = time.time()
            if verbosity > 1: progbar = tf.keras.utils.Progbar(batch_count)
            for (batch, (original_x)) in enumerate(train_dataset):
                processed_x = self.preprocess_input(original_x)
                gradients, loss = self.compute_gradients(original_x, processed_x)
                train_loss(loss)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
                if verbosity > 1: progbar.update(batch+1)
            end_time = time.time()
        
            for original_x in val_dataset:
                processed_x = self.preprocess_input(original_x)
                loss, _ = self.compute_loss(original_x, processed_x)
                val_loss.update_state(loss)

            with log_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                tf.summary.scalar('val_loss', val_loss.result(), step=epoch)

            if verbosity >= 1:
                print(f"Epoch {epoch}, Train_Loss {train_loss.result()}, Val_Loss {val_loss.result()} Train_Time {end_time - start_time}")

            if callbacks:
                for callback in callbacks: callback(self)
            
            train_loss.reset_states()
            val_loss.reset_states()
        
        self.save_weights(checkpoint_dir)
            
class GenerativeAutoEncoder(AutoEncoder):
    """ An autoencoder that encodes data as some distribution. Supports sampling. """
    def __init__(self, encoder, decoder, preprocessing_steps=[], call_func='vae', loss_funcs=[reconstruction()], latent_dim=32):
        super().__init__( encoder, decoder, preprocessing_steps=preprocessing_steps, call_func=call_func, loss_funcs=loss_funcs)
        self.latent_dim = latent_dim # @TODO store in encoder?
    
    def sample(self, eps):
        """ Returns a sampling from the distribution, given a code vector."""
        logits = self.decode(eps)
        probs = tf.sigmoid(logits)
        return probs
        
    def sample_random(self):
        """ Returns a random sampling from the distribution. """
        eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.sample(eps)
