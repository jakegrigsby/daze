import argparse

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import *
from deepzip.nets.decoders import *


def train_encoder(encoder_type, dataset_type, run_name):
    # Use dataset to infer encoder, decoder
    if dataset_type in ["cifar", "cifar10"]:
        dataset = dz.data.cifar10
        encoder = EncoderCifar10()
        decoder = DecoderCifar10()
    else:
        raise ValueError(f"Invalid dataset code {datset_str}. Options are: 'cifar'...")

    # Select algorithm type
    if encoder_type == "default":
        model = dz.core.AutoEncoder(encoder, decoder)
    elif encoder_type == "vae":
        model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
    elif encoder_type == "denoising":
        model = dz.recipes.DenoisingAutoEncoder(encoder, decoder, 0.15)
    elif encoder_type == "sparse":
        model = dz.recipes.SparseAutoEncoder(encoder, decoder)
    elif encoder_type == "contractive":
        model = dz.recipes.ContractiveAutoEncoder(encoder, decoder)
    else:
        raise ValueError("Invalid autoencoder type {}".format(encoder_type))

    # Load data, train model
    x_train, x_val = dataset.load()
    model.train(x_train, x_val, epochs=10, experiment_name=run_name, verbosity=2)


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-type", default="default", type=str, help="model type, like vae or default"
    )
    parser.add_argument("-dataset", default="cifar", type=str)
    parser.add_argument("-name", default="testrun", type=str)
    args = parser.parse_args()
    train_encoder(args.type, args.dataset, args.name)


if __name__ == "__main__":
    main()
