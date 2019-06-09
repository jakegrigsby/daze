import tensorflow as tf

def load():
    """
    Will take a while to run (the first time)
    Returns: tuple of numpy arrays (x_train, y_train), (x_test, y_test)
    """
    return tf.keras.datasets.cifar10.load_data()


if __name__ == "__main__":
    train, test = load()
    import matplotlib.pyplot as plt
    plt.imshow(train[0][0,...])
    plt.show()