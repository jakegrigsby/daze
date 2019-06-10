import tensorflow as tf

def load():
    """
    Will take a while to run (the first time)
    """
    (x_train, _), (x_val, _) = tf.keras.datasets.cifar10.load_data()
    return x_train, x_val


if __name__ == "__main__":
    train, test = load()
    import matplotlib.pyplot as plt
    plt.imshow(train[0][0,...])
    plt.show()