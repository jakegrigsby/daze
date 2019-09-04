
import tensorflow as tf
# Currently, we still require the nightly build of tensorflow_probability
import tensorflow_probability as tfp


from .math import *
from .tracing import trace_graph
from . import preprocessing

def probabilistic_encode_decode():
    @trace_graph
    def _probabilistic_encode_decode(model, original_x, x):
        """VAE-style forward pass.

        Args:
            model (daze.model.Model) : Model to use for this prediction.
            original_x (tf.Tensor) : Original X (without preprocessing steps).
            x (tf.Tensor) : Preprocessed X.
        
        Returns:
            forward_pass_dict containing: original_x, x, mean, sigma, z, x_hat.
        """
        mean, sigma = tf.split(model.encode(x), num_or_size_splits=2, axis=1)
        q_z = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        z = q_z.sample()
        x_hat = model.decode(mean)
        return {
            "original_x": original_x,
            "x": x,
            "mean": mean,
            "sigma": sigma,
            "z": z,
            "x_hat": x_hat,
        }
    return _probabilistic_encode_decode


def standard_encode_decode():
    @trace_graph
    def _standard_encode_decode(model, original_x, x):
        """Autoencoder-style forward pass.

        Args:
            model (daze.model.Model) : Model to use for this prediction.
            original_x (tf.Tensor) : Original X (without preprocessing steps).
            x (tf.Tensor) : Preprocessed X.
        
        Returns:
            forward_pass_dict containing: original_x, x, h, x_hat
        """
        h = model.encode(x)
        x_hat = model.decode(h)
        return {"original_x": original_x, "x": x, "h": h, "x_hat": x_hat}
    return _standard_encode_decode


def generative_adversarial():
    @trace_graph
    def _generative_adversarial(model, original_x, x):
        noise = tf.random.normal([x.shape[0], model.noise_dim])
        generated_images = model.generate(noise)
        real_features, real_output = model.discriminate(x)
        fake_features, fake_output = model.discriminate(generated_images)
        return {
                "generated_images" : generated_images,
                "real_features" : real_features,
                "real_output" : real_output,
                "fake_features" : fake_features,
                "fake_output" : fake_output,
        }
    return _generative_adversarial


def generative_adversarial_instance_noise(gamma_start, gamma_end, steps_annealed, mean=0., std=1.):
    current_gamma = gamma_start
    @trace_graph
    def _generative_adversarial_instance_noise(model, x, original_x):
        nonlocal current_gamma
        noise = tf.random.normal([x.shape[0], model.noise_dim])
        generated_images = model.generate(noise)
        current_gamma = max(current_gamma - (gamma_start - gamma_end)/steps_annealed, gamma_end)
        _gaussian_noise = preprocessing.gaussian_noise(mean, std, current_gamma)
        generated_images = _gaussian_noise(generated_images)
        x = _gaussian_noise(x)
        real_features, real_output = model.discriminate(x)
        fake_features, fake_output = model.discriminate(generated_images)
        return {
                "generated_images" : generated_images,
                "real_features" : real_features,
                "real_output" : real_output,
                "fake_features" : fake_features,
                "fake_output" : fake_output,
        }
    return _generative_adversarial_instance_noise


