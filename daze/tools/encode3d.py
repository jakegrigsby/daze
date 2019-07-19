import argparse

import numpy as np

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder, MnistDecoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", type=str)
    parser.add_argument("-output", type=str)
    parser.add_argument("-dataset", type=str, choices=["mnist", "cifar", "cifar10"])
    args = parser.parse_args()

    if args.dataset in ["cifar", "cifar10"]:
        dataset = dz.data.cifar10
        Decoder = CifarDecoder
    elif args.dataset in ["mnist"]:
        dataset = dz.data.mnist
        Decoder = MnistDecoder

    model = dz.Model(ConvolutionalEncoder(latent_dim=3), Decoder())
    model.load_weights(args.weights)

    data, _ = dataset.load(dtype="f")
    data /= 255.0

    encodings = model.get_batch_encodings(data)
    np.savetxt(args.output, encodings)
