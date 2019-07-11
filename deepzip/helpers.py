""" Helper functions for internal things. Not exposed as part of
    the public API. """
from functools import wraps

import numpy as np
import tensorflow as tf

import deepzip as dz

###################
# Misc Math Funcs #
###################

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


def softmax(x):
    return 1.0 * x / tf.math.reduce_sum(x)


def kl_divergence(a, b):
    return (
        a * tf.math.log(a)
        - a * tf.math.log(b)
        + (1 - a) * tf.math.log(1 - a)
        - (1 - a) * tf.math.log(1 - b)
    )


def sample(model, eps):
    logits = model.decode(eps)
    probs = tf.sigmoid(logits)
    return probs


def sample_random(model):
    eps = tf.random.normal(shape=(100, model.latent_dim))
    return model.sample(eps)

#####################
# Graph Compilation #
#####################

_TRACE_GRAPHS = True
_TRACE_RECORD = {}

def _add_to_trace_record(func):
    global _TRACE_RECORD
    name = func.__name__
    if name in _TRACE_RECORD:
        count = _TRACE_RECORD[name]
        _TRACE_RECORD[name] += 1
        if count == 10: retrace_indicator(name)
    else:
        _TRACE_RECORD[name] = 1 

def reset_trace_record():
    global _TRACE_RECORD
    _TRACE_RECORD = {}

def trace_graph(func):
    if _TRACE_GRAPHS:
        @wraps(func)
        @tf.function
        def func_with_trace_count(*args, **kwargs):
            _add_to_trace_record(func)
            return func(*args, **kwargs)
        return func_with_trace_count
    else:
        return func

def is_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        print("{}({}) uses tf.while_loop.".format(
        f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        print("{}({}) uses tf.data.Dataset.reduce.".format(
        f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) gets unrolled.".format(
        f.__name__, ', '.join(map(str, args))))

def retrace_indicator(func_name):
    print(f"\nWarning: {func_name} is being traced repeatedly. See "
        "https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function "
        "for more information.")