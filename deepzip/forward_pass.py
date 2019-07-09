import tensorflow as tf

from . import helpers


def probabalistic_encode_decode(model, original_x, x):
    mean, logvar = tf.split(model.encode(x), num_or_size_splits=2, axis=1)
    z = helpers.reparameterize(mean, logvar)
    x_hat = model.decode(z)
    return {
        "original_x": original_x,
        "x": x,
        "mean": mean,
        "logvar": logvar,
        "z": z,
        "x_hat": x_hat,
    }


def standard_encode_decode(model, original_x, x):
    h = model.encode(x)
    x_hat = model.decode(h)
    return {"original_x": original_x, "x": x, "h": h, "x_hat": x_hat}
