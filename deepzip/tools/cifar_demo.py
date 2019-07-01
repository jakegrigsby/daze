import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import deepzip as dz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', type=str)
    args = parser.parse_args()

    weights = os.path.join('../data/', args.weights)

    model = dz.recipes.DenoisingAutoEncoder(dz.nets.encoders.EasyEncoder(), dz.nets.decoders.EasyDecoder(), .15)
    model.load_weights(weights)

    _, cifar_val = dz.data.cifar10.load()
    np.random.shuffle(cifar_val)

    test_images = cifar_val[:5,...]

    rows = 5
    columns = 2
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        img = test_images[row,...]
        axarr[row,0].imshow(img)
        x_hat = model.predict(np.expand_dims(img, 0))
        axarr[row,1].imshow(np.squeeze(x_hat))
    plt.show()