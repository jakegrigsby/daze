
import tensorflow as tf
import numpy as np


def load(size=None):
    (x_train, _), (x_val, _) = tf.keras.datasets.boston_housing.load_data()
    size = x_train.shape[0] if not size else size
    return x_train[:size], x_val[:size]

if __name__ == "__main__":
    train, val = load()
    print("Loaded Boston Housing Dataset...")
    print(f"Training Set shape: {train.shape}")
    print(f"Validation Set shape: {val.shape}")