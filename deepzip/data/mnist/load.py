import tensorflow as tf

import deepzip as dz

def load(size=None, dtype=None):
    (x_train, _) , (x_val, _) = tf.keras.datasets.mnist.load_data()
    size = x_train.shape[0] if not size else size
    x_train, x_val = x_train[:size], x_val[:size]
    if dtype:
        dtype = dz.data.utils.parse_dtype(dtype)
        x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    return x_train, x_val

if __name__ == "__main__":
    train, val = load()
    import matplotlib.pyplot as plt
    print("Loaded MNIST Dataset...")
    print(f"Training Set shape: {train.shape}")
    print(f"Validation Set shape: {val.shape}")
    plt.imshow(train[0])
    plt.show()
