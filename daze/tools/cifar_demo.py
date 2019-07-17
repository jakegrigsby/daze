import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import deepzip as dz
from deepzip.nets.encoders import Encoder_32x32
from deepzip.nets.decoders import Decoder_32x32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", type=str)
    args = parser.parse_args()

    model = dz.recipes.DenoisingAutoEncoder(Encoder_32x32(), Decoder_32x32(), gamma=0.1)
    model.load_weights(args.weights)

    _, cifar_val = dz.data.cifar10.load(dtype="f")
    cifar_val /= 255.0
    np.random.shuffle(cifar_val)

    test_images = cifar_val[:5, ...]

    rows = 5
    columns = 2
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        img = test_images[row, ...]
        axarr[row, 0].imshow(img)
        x_hat = model.predict(np.expand_dims(img, 0))
        axarr[row, 1].imshow(np.squeeze(x_hat))
    plt.show()
