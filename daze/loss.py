
import numpy as np
import tensorflow as tf
# Currently, we still require the nightly build of tensorflow_probability
import tensorflow_probability as tfp

from .math import *
from .tracing import trace_graph, TRACE_GRAPHS
from .enforce import *

mse = tf.keras.losses.mean_squared_error


def kl(beta):
    """KL Divergence Loss Term

    Args:
        beta (float) : coefficient for this terms' contribution to overall
            loss function.
    """
    @trace_graph
    @vae_compatible
    def _kl(**forward_pass):
        mean = forward_pass["mean"]
        sigma = forward_pass["sigma"]
        z = forward_pass["z"]
        
        q_z = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        p_z = tfp.distributions.MultivariateNormalDiag(
          loc=[0.0] * z.shape[-1], scale_diag=[1.0] * z.shape[-1]
          ) # @TODO: pre-compute this normal dist. and store
        
        kl_div = tfp.distributions.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
        
        return beta * latent_loss

    return _kl


def contractive(coeff):
    """Contractive Loss Term

    Note:
        This loss function can't be compiled (for now) because of the way it calculates
        gradients as part of the inner loop.

    Args:
        coeff (float) : coefficient for this terms' contribution to the overall
            loss function.
    """
    # this can't be compiled into a tf.function because of its gradient calculation
    if TRACE_GRAPHS:
        raise ValueError(
            "Autograph tracing not supported for contractive loss. Set dz_trace_graphs environment variable to false:"
            "\t`$ export dz_trace_graphs=False`"
        )
    @ae_compatible
    def _contractive(**forward_pass):
        h = forward_pass["h"]
        x = forward_pass["x"]
        tape = forward_pass["tape_container"].tape
        dh_dx = tape.gradient(h, x)
        frob_norm = tf.norm(dh_dx)
        return coeff * frob_norm

    return _contractive


def denoising_reconstruction():
    """Mean Squared Error between reconstruction and the original x (before
        preprocessing).
    """
    @trace_graph
    @encoder_decoder_compatible
    def _denoising(**forward_pass):
        original_x = forward_pass["original_x"]
        x_hat = forward_pass["x_hat"]
        return mse(x_hat, original_x)

    return _denoising


def reconstruction():
    """Mean Squared Error between the reconstruction and true (preprocessed) input.
    """
    @trace_graph
    @encoder_decoder_compatible
    def _reconstruction(**forward_pass):
        x = forward_pass["x"]
        x_hat = forward_pass["x_hat"]
        return mse(x, x_hat)

    return _reconstruction


def latent_l1(gamma):
    """Loss term based on the L1 distance of the latent space.

    Args:
        gamma (float) : coefficient for this terms' contribution to the overall
            loss function.
    """
    @trace_graph
    @ae_compatible
    def _latent_l1(**forward_pass):
        h = forward_pass["h"]
        return gamma * tf.math.reduce_sum(tf.math.abs(h))

    return _latent_l1


def sparsity(rho, beta):
    """Sparsity loss term.

    Args:
        rho (float) : the target sparsity value (~.01)
        beta (float) : coefficient for this terms' contribution to the overall loss function.
    """
    rho = tf.constant(rho)

    @trace_graph
    @ae_compatible
    def _sparsity(**forward_pass):
        h = forward_pass["h"]
        rho_hat = tf.reduce_mean(h, axis=0)
        return beta * tf.keras.losses.kld(rho, rho_hat)

    return _sparsity

def maximum_mean_discrepancy():
    """Maximum Mean Discrepancy

    Paper: https://arxiv.org/pdf/1706.02262.pdf
    """
    def compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    true_samples = None
    
    def sample(dim):
        nonlocal true_samples
        true_samples = tf.random.normal(tf.stack([200, dim]))

    @trace_graph
    @vae_compatible
    def _maximum_mean_discrepancy(**forward_pass):
        z = forward_pass["z"]        
        if not isinstance(true_samples, tf.Tensor): sample(z.shape[1])
        x = true_samples
        x_kernel = compute_kernel(x, x)
        z_kernel = compute_kernel(z, z)
        xz_kernel = compute_kernel(x, z)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(z_kernel) - 2 * tf.reduce_mean(xz_kernel)
    
    return _maximum_mean_discrepancy

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def vanilla_discriminator_loss():
    @trace_graph
    @discriminator_compatible
    def _vanilla_discriminator_loss(**forward_pass):
        real_output = forward_pass["real_output"]
        fake_output = forward_pass["fake_output"]
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    return _vanilla_discriminator_loss

def vanilla_generator_loss():
    @trace_graph
    @generator_compatible
    def _vanilla_generator_loss(**forward_pass):
        fake_output = forward_pass["fake_output"]
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    return _vanilla_generator_loss
 
def one_sided_label_smoothing(smoothing_val=.9):
    @trace_graph
    @discriminator_compatible
    def _one_sided_label_smoothing(**forward_pass):
        real_output = forward_pass["real_output"]
        fake_output = forward_pass["fake_output"]
        real_loss = cross_entropy(tf.ones_like(real_output)*smoothing_val, real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    return _one_sided_label_smoothing

def feature_matching(gamma=1.):
    @trace_graph
    @generator_compatible
    def _feature_matching(**forward_pass):
        real_features = forward_pass["real_features"]
        fake_features = forward_pass["fake_features"]
        return gamma * mse(real_features, fake_features)
    return _feature_matching

          
