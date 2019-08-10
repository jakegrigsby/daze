import tensorflow as tf

from .math import *
from .tracing import trace_graph

# Currently, we still require the nightly build of tensorflow_probability
import tensorflow_probability as tfp


@trace_graph
def probabalistic_encode_decode(model, original_x, x):
    mean, sigma = tf.split(model.encode(x), num_or_size_splits=2, axis=1)
    q_z = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
    z = q_z.sample()
    x_hat = model.decode(z)
    return {
        "original_x": original_x,
        "x": x,
        "mean": mean,
        "sigma": sigma,
        "z": z,
        "x_hat": x_hat,
    }


@trace_graph
def standard_encode_decode(model, original_x, x):
    h = model.encode(x)
    x_hat = model.decode(h)
    return {"original_x": original_x, "x": x, "h": h, "x_hat": x_hat}
