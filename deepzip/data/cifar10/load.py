import tensorflow as tf
import numpy as np

def load():
    """
    Will take a while to run (the first time)
    """
    (x_train, _), (x_val, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_val = x_val.astype(np.float32) / 255
    return x_train, x_val


if __name__ == "__main__":
    train, test = load()
    import matplotlib.pyplot as plt
    plt.imshow(train[0][0,...])
    plt.show()