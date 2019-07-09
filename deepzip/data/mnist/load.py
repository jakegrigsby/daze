import tensorflow as tf

def load(size=None):
    (x_train, _) , (x_val, _) = tf.keras.datasets.mnist.load_data()
    size = x_train.shape[0] if not size else size
    return x_train[:size], x_val[:size]

if __name__ == "__main__":
    train, val = load()
    import matplotlib.pyplot as plt
    print("Loaded MNIST Dataset...")
    print(f"Training Set shape: {train.shape}")
    print(f"Validation Set shape: {val.shape}")
    plt.imshow(train[0])
    plt.show()
