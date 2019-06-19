import numpy as np
import tensorflow as tf

def random_mask(destruction_coeff, seed=None):
    def _random_mask(input_batch):
        """
        Random destruction of data.
        A fraction of the input tensor (determined by destruction_coeff) is
        randomly set to 0.
        """
        total_size = tf.size(input_batch).numpy()
        num_set_zero = int(total_size*destruction_coeff)
        idxs = np.random.choice(np.arange(total_size), num_set_zero)
        mask = np.zeros(total_size)
        mask[idxs] = 1.
        mask = tf.dtypes.cast(tf.reshape(mask, input_batch.shape), tf.float32)
        return input_batch * mask
    return _random_mask

def gaussian_noise(input_batch, mean, std, seed=None):
    """
    Inject random gaussian noise with given mean and std.
    """
    noise = tf.random.normal(input_batch.shape, mean, std, seed=seed)
    return input_batch + noise

def random_mask(destr):
    def _random_mask(input_batch, seed=None):
        total_size = tf.size(input_batch).numpy()
        num_set_zero = int(total_size*destruction_coeff)
        idxs = np.random.choice(np.arange(total_size), num_set_zero)
        mask = np.zeros(total_size)
        mask[idxs] = 1.
        mask = tf.dtypes.cast(tf.reshape(mask, input_batch.shape), tf.float32)
        return input_batch * mask
    return functools.partial(sum2, 1)