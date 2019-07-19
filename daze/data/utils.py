import math

import numpy as np
import tensorflow as tf


def get_byte_count(s):
    idx = -1
    if not s[idx].isnumeric():
        return None
    while s[idx - 1 :].isnumeric() and idx > -len(s):
        idx -= 1
    return int(s[idx:])


def parse_dtype(s):
    if not type(s) == str:
        # s is already in correct numpy dtype form
        return s
    err = f"Datatype string code not recognized: {s}"
    byte_count = get_byte_count(s)
    if s[0] == "f":
        # float
        if not byte_count:
            return np.float32
        elif byte_count in (16, 32, 64):
            return eval(f"np.float{byte_count}")
        else:
            raise ValueError(err)
    elif s[0] == "i":
        # int
        if not byte_count:
            return np.int32
        elif byte_count in (8, 16, 32, 64):
            return eval(f"np.int{byte_count}")
        else:
            raise ValueError(err)
    elif s[0] == "u":
        # unsigned int
        if not byte_count:
            return np.uint32
        elif byte_count in (8, 16, 32, 64):
            return eval(f"np.uint{byte_count}")
        else:
            raise ValueError(err)
    else:
        raise ValueError(err)


def convert_np_to_tf(np_dataset, batch_size, return_batch_count=False):
    dataset_size = np_dataset.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(np_dataset)
    dataset = dataset.shuffle(dataset_size + 1).batch(batch_size)
    if return_batch_count:
        batch_count = math.ceil(dataset_size / batch_size)
        return dataset, batch_count
    return dataset

def load_from_file(filepath):
    return np.loadtxt(filepath)

def random_shuffle(*arrays):
    permutation = np.random.permutation(arrays[0].shape[0])
    shuffled = []
    for array in arrays:
        shuffled.append(array[permutation])
    return tuple(shuffled)

def get_sample_count(*arrays):
    if all(array.shape[0] == arrays[0].shape[0] for array in arrays):
        sample_count = arrays[0].shape[0]
    else:
        sample_count = None
    return sample_count

def train_val_split(*arrays, split_val=.3):
    sample_count = get_sample_count(*arrays)
    if not sample_count:
        raise Exception("Batch Axis inconsistent. All input arrays must have first axis of equal length.")
    arrays = random_shuffle(*arrays)
    split_idx = math.floor(sample_count * split_val)
    train_set = [array[split_idx:] for array in arrays]
    val_set = [array[:split_idx] for array in arrays]
    return train_set, val_set