import argparse

import matplotlib.pyplot as plt
import numpy as np

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder, MnistDecoder


def reshape_for_prediction(img):
    if not img.shape[0] == 1:
        # insert batch dimension
        img = np.expand_dims(img, 0)
    if img.ndim == 3:
        # insert channel dimension
        img = np.expand_dims(img, -1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", type=str)
    parser.add_argument("-dataset", type=str)
    parser.add_argument("-latent_size", type=int, default=32)
    args = parser.parse_args()

    if args.dataset in ["cifar", "cifar10"]:
        dataset, _ = dz.data.cifar10.load(dtype="f")
        Decoder = CifarDecoder
    elif args.dataset in ["mnist"]:
        dataset, _ = dz.data.mnist.load(dtype="f")
        dataset = np.squeeze(dataset)
        Decoder = MnistDecoder

    model = dz.Model(ConvolutionalEncoder(latent_dim=args.latent_size), Decoder())
    model.load_weights(args.weights)

    dataset /= 255.0
    np.random.shuffle(dataset)

    test_images = dataset[:5, ...]

    rows = 5
    columns = 2
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        img = test_images[row, ...]
        axarr[row, 0].imshow(img)
        x_hat = model.predict(reshape_for_prediction(img))
        axarr[row, 1].imshow(np.squeeze(x_hat))
    plt.show()
