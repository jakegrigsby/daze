import os
import time
import math
import functools

import numpy as np
import tensorflow as tf

from . import loss
from . import forward_pass
from .tracing import reset_trace_record, trace_graph
from . import data

class DZModel:
    def __init__(self, preprocessing_steps=[]):
        self.preprocessing_steps = preprocessing_steps

    def make_tape_container(self):
        """
        Trick the autograph system into only tracing each function once by wrapping python
        objects in a superobject with a constant id. We use this to pass gradient tapes
        to contractive loss.
        """
        class _TapeContainer:
            def __init__(self):
                self.tape = None
        return _TapeContainer()

    def preprocess_input(self, x):
        for func in self.preprocessing_steps:
            x = func(x)
        return x

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

    def apply_gradients(self, optimizer, gradients, variables):
        """ Applies the gradients to the optimizer. """
        optimizer.apply_gradients(zip(gradients, variables))

class AutoEncoder(DZModel):
    def __init__(
        self,
        encoder,
        decoder,
        preprocessing_steps=[],
        forward_pass_func=forward_pass.standard_encode_decode,
        loss_funcs=[loss.reconstruction()],
    ):
        super().__init__(preprocessing_steps)
        self.encoder, self.decoder = encoder, decoder
        self.call = functools.partial(forward_pass_func, self)
        self.loss_funcs = loss_funcs
        self.tape_container = self.make_tape_container()

    def compute_loss(self, original_x, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            forward_pass = self.call(original_x, x)
            self.tape_container.tape = tape
            forward_pass["tape_container"] = self.tape_container
            loss = 0
            for loss_func in self.loss_funcs:
                loss += loss_func(**forward_pass)
        return loss, tape

    def predict(self, x):
        return self.call(x, x)["x_hat"]

    def get_batch_encodings(self, x):
        if not isinstance(x, tf.data.Dataset):
            if isinstance(x, np.ndarray):
                x = data.utils.convert_np_to_tf(x, 32)
            else:
                raise ValueError(f"Dataset of type {type(x)} not supported.")
        out_data = None
        for batch in x:
            encoding = self.encode(batch)
            if isinstance(out_data, tf.Tensor):
                out_data = tf.concat((out_data, encoding), axis=0)
            else:
                out_data = encoding
        return out_data

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

    def compute_gradients(self, original_x, x):
        """ Computes gradient of custom loss function.
        """
        loss, tape = self.compute_loss(original_x, x)
        grad = tape.gradient(loss, self.trainable_variables)
        return grad, loss

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs=10,
        save_path="saves",
        verbosity=1,
        callbacks=None,
    ):
        """ Trains the model for a given number of epochs (iterations on a dataset).
        """
        reset_trace_record()
        log_dir, checkpoint_dir = self.init_logging(save_path)

        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)

        batch_count = 0
        for x in train_dataset:
            batch_count += 1

        for epoch in range(epochs):
            if verbosity > 1: progbar = tf.keras.utils.Progbar(batch_count)
            start_time = time.time()
            for (batch, (original_x)) in enumerate(train_dataset):
                processed_x = self.preprocess_input(original_x)
                gradients, loss = self.compute_gradients(original_x, processed_x)
                train_loss(loss)
                self.apply_gradients(optimizer, gradients, self.trainable_variables)
                if verbosity > 1 and batch_count:
                    progbar.update(batch + 1)
                if callbacks:
                    batch_dict = {
                    "type":"batch",
                    "gradients": gradients,
                    "current_step": (epoch*batch_count)+batch,
                    "log_dir": log_dir,
                    }
                    for callback in callbacks:
                        callback(self, **batch_dict)
                
            end_time = time.time()
            
            for original_x in val_dataset:
                loss, _ = self.compute_loss(original_x, original_x)
                val_loss(loss)

            if verbosity >= 1:
                print(
                    f"Epoch {epoch}, Train_Loss {train_loss.result()}, Val_Loss {val_loss.result()} Train_Time {end_time - start_time}"
                )
            
            if callbacks:
                epoch_dict = {
                "type":"epoch",
                "current_epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_start_time": start_time,
                "epoch_end_time": end_time,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                }
                for callback in callbacks:
                    callback(self, **epoch_dict)

            train_loss.reset_states()
            val_loss.reset_states()

        self.save_weights(checkpoint_dir)


class GAN(DZModel):
    def __init__(
        self,
        generator,
        discriminator,
        noise_dim,
        preprocessing_steps=[],
        forward_pass_func=forward_pass.generative_adversarial,
        generator_loss=[loss.vanilla_generator_loss()],
        discriminator_loss=[loss.vanilla_discriminator_loss()],
    ):
        super().__init__(preprocessing_steps)
        self.generator, self.discriminator = generator, discriminator
        self.call = functools.partial(forward_pass_func, self)
        self.generator_loss_funcs = generator_loss
        self.noise_dim = noise_dim
        self.discriminator_loss_funcs = discriminator_loss

        self.gen_tape_container = self.make_tape_container()
        self.disc_tape_container = self.make_tape_container()

    def compute_loss(self, original_x, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            forward_pass = self.call(original_x, x)
            self.gen_tape_container.tape = gen_tape
            self.disc_tape_container.tape = disc_tape
            forward_pass["gen_tape_container"] = self.gen_tape_container
            forward_pass["disc_tape_container"] = self.disc_tape_container
            gen_loss = 0
            for loss_func in self.generator_loss_funcs:
                gen_loss += loss_func(**forward_pass)
            disc_loss = 0
            for loss_func in self.discriminator_loss_funcs:
                disc_loss += loss_func(**forward_pass)
        return gen_loss, gen_tape, disc_loss, disc_tape

    @trace_graph
    def generate(self, x):
        return self.generator(x)
    
    @trace_graph
    def discriminate(self, x):
        return self.discriminator(x)

    @property
    def weights(self):
        return [self.generator.get_weights(), self.discriminator.get_weights()]

    @weights.setter
    def weights(self, new_weights):
        self.generator.set_weights(new_weights[0])
        self.discriminator.set_weights(new_weights[1])

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.generator.save_weights(os.path.join(path, "generator"))
        self.discriminator.save_weights(os.path.join(path, "discriminator"))

    def load_weights(self, path):
        self.generator.load_weights(os.path.join(path, "generator"))
        self.discriminator.load_weights(os.path.join(path, "discriminator"))
        
    def compute_gradients(self, original_x, x):
        """ Computes gradient of custom loss function.
        """
        gen_loss, gen_tape, disc_loss, disc_tape = self.compute_loss(original_x, x)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        return gen_grads, gen_loss, disc_grads, disc_loss

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs=10,
        save_path="saves",
        verbosity=1,
        callbacks=None,
    ):
        """ Trains the model for a given number of epochs (iterations on a dataset).
        """
        reset_trace_record()
        log_dir, checkpoint_dir = self.init_logging(save_path)

        gen_optimizer = tf.keras.optimizers.Adam()
        disc_optimizer = tf.keras.optimizers.Adam()

        train_loss_gen = tf.keras.metrics.Mean("generator_train_loss", dtype=tf.float32)
        train_loss_disc = tf.keras.metrics.Mean("discriminator_train_loss", dtype=tf.float32)

        val_loss_gen = tf.keras.metrics.Mean("generator_val_loss", dtype=tf.float32)
        val_loss_disc = tf.keras.metrics.Mean("discriminator_val_loss", dtype=tf.float32)

        batch_count = 0
        for x in train_dataset:
            batch_count += 1

        for epoch in range(epochs):
            if verbosity > 1: progbar = tf.keras.utils.Progbar(batch_count)
            start_time = time.time()
            for (batch, (original_x)) in enumerate(train_dataset):
                processed_x = self.preprocess_input(original_x)
                gen_grads, gen_loss, disc_grads, disc_loss = self.compute_gradients(original_x, processed_x)
                train_loss_gen(gen_loss)
                train_loss_disc(disc_loss)
                self.apply_gradients(gen_optimizer, gen_grads, self.generator.trainable_variables)
                self.apply_gradients(disc_optimizer, disc_grads, self.discriminator.trainable_variables)
                if verbosity > 1 and batch_count:
                    progbar.update(batch + 1)
                if callbacks:
                    batch_dict = {
                    "type":"batch",
                    "gradients": gen_grads,
                    "current_step": (epoch*batch_count)+batch,
                    "log_dir": log_dir,
                    }
                    for callback in callbacks:
                        callback(self, **batch_dict)
                
            end_time = time.time()
            
            for original_x in val_dataset:
                gen_loss, _, disc_loss, _ = self.compute_loss(original_x, original_x)
                val_loss_gen(gen_loss)
                val_loss_disc(disc_loss)

            if verbosity >= 1:
                print(
                        f"Epoch {epoch}, Train_Loss (Gen: {train_loss_gen.result()}, Disc: {train_loss_disc.result()}), Val_Loss (Gen: {val_loss_gen.result()}, Disc: {val_loss_disc.result()}) Train_Time {end_time - start_time}"
                )
            
            if callbacks:
                epoch_dict = {
                "type":"epoch",
                "current_epoch": epoch,
                "train_loss": train_loss_gen,
                "val_loss": val_loss_gen,
                "epoch_start_time": start_time,
                "epoch_end_time": end_time,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                }
                for callback in callbacks:
                    callback(self, **epoch_dict)

            train_loss_gen.reset_states()
            train_loss_disc.reset_states()
            val_loss_gen.reset_states()
            val_loss_disc.reset_states()

        self.save_weights(checkpoint_dir)
