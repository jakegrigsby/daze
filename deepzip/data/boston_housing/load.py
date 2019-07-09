
import tensorflow as tf
import numpy as np

import deepzip as dz

def load(size=None, dtype=None):
    (x_train, _), (x_val, _) = tf.keras.datasets.boston_housing.load_data()
    size = x_train.shape[0] if not size else size
    x_train, x_val = x_train[:size], x_val[:size]
    if dtype:
        dtype = dz.data.utils.parse_dtype(dtype)
        x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    return x_train, x_val


if __name__ == "__main__":
    train, val = load()
    print("Loaded Boston Housing Dataset...")
    print(f"Training Set shape: {train.shape}")
    print(f"Validation Set shape: {val.shape}")