import os
import time
import math
import functools

import numpy as np
import tensorflow as tf

from .loss import reconstruction
from .helpers import trace_graph, reset_trace_record
from . import forward_pass

class Model(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        preprocessing_steps=[],
        forward_pass_func=forward_pass.standard_encode_decode,
        loss_funcs=[reconstruction()],
    ):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.preprocessing_steps = preprocessing_steps
        self.call = functools.partial(forward_pass_func, self)
        self.loss_funcs = loss_funcs

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
                loss += loss_func(**forward_pass)
        return loss, tape

    def predict(self, x):
        return self.call(x, x)['x_hat']

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

    @trace_graph
    def encode(self, x):
        return self.encoder(x)

    @trace_graph
    def decode(self, h):
        return self.decoder(h)

    def init_logging(self, save_path):
        """ Sets up log directories for training.
        """
        save_path = os.path.join(os.getcwd(), save_path)
        # get unique number for this run
        i = 0
        while os.path.exists(save_path + f"_{i}"):
            i += 1
        save_path += f"_{i}"
        # Setup checkpoints and logging
        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir)
        log_dir = os.path.join(save_path, "logs")
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

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs=10,
        save_path="saves",
        verbosity=1,
        callbacks=None,
        batch_size=64,
    ):
        """ Trains the model for a given number of epochs (iterations on a dataset).
        """
        reset_trace_record()
        log_dir, checkpoint_dir = self.init_logging(save_path)

        optimizer = tf.keras.optimizers.Adam()

        train_set_size = train_dataset.shape[0]
        batch_count = math.ceil(train_set_size / batch_size)
        train_dataset = self.create_dataset(train_dataset, batch_size)
        val_dataset = self.create_dataset(val_dataset, batch_size)

        train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)
        for epoch in range(epochs):
            start_time = time.time()
            if verbosity > 1:
                progbar = tf.keras.utils.Progbar(batch_count)
            for (batch, (original_x)) in enumerate(train_dataset):
                processed_x = self.preprocess_input(original_x)
                gradients, loss = self.compute_gradients(original_x, processed_x)
                train_loss(loss)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
                if verbosity > 1:
                    progbar.update(batch + 1)
            end_time = time.time()

            for original_x in val_dataset:
                loss, _ = self.compute_loss(original_x, original_x)
                val_loss(loss)

            if verbosity >= 1:
                print(
                    f"Epoch {epoch}, Train_Loss {train_loss.result()}, Val_Loss {val_loss.result()} Train_Time {end_time - start_time}"
                )

            train_dict = {
                "current_epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_start_time": start_time,
                "epoch_end_time": end_time,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
            }
            if callbacks:
                for callback in callbacks:
                    callback(self, **train_dict)

            train_loss.reset_states()
            val_loss.reset_states()

        self.save_weights(checkpoint_dir)
