import argparse
import sys

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import *
from deepzip.nets.decoders import *
from deepzip.callbacks import checkpoints, tensorboard
from deepzip.preprocessing import basic_image_normalize


def train_encoder(model_type, dataset_type, epochs):
    # Use dataset to infer encoder, decoder
    if dataset_type in ["cifar", "cifar10"]:
        dataset = dz.data.cifar10
    elif dataset_type in ["mnist"]:
        dataset = dz.data.mnist
    else:
        raise ValueError(f"Invalid dataset code {datset_type}. Options are: 'cifar'...")

    x_train, x_val = dataset.load(dtype="f")
    x_train /= 255
    x_train = dz.data.utils.convert_np_to_tf(x_train, 32)
    x_val /= 255
    x_val = dz.data.utils.convert_np_to_tf(x_val, 32)
    encoder = Encoder_32x32()
    decoder = Decoder_32x32()
    callbacks = [checkpoints(1), tensorboard()]

    # Select algorithm type
    if model_type == "default":
        model = dz.Model(Encoder_32x32(), Decoder_32x32())
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/default",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "vae":
        model = dz.recipes.VariationalAutoEncoder(Encoder_32x32(), Decoder_32x32())
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/vae",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "denoising":
        model = dz.recipes.DenoisingAutoEncoder(
            Encoder_32x32(), Decoder_32x32(), gamma=0.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/denoising",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "l1sparse":
        model = dz.recipes.L1SparseAutoEncoder(
            Encoder_32x32(), Decoder_32x32(), gamma=0.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/l1sparse",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "contractive":
        model = dz.recipes.ContractiveAutoEncoder(
            Encoder_32x32(), Decoder_32x32(), gamma=0.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/contractive",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "bvae":
        model = dz.recipes.BetaVariationalAutoEncoder(
            Encoder_32x32(), Decoder_32x32(), beta=1.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/bvae",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "klsparse":
        model = dz.recipes.KlSparseAutoEncoder(
            Encoder_32x32(), Decoder_32x32(), rho=0.01, beta=0.1
        )
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/klsparse",
            epochs=epochs,
            verbosity=2,
        )
    else:
        raise ValueError("Invalid autoencoder type {}".format(model_type))


def main():
    parser = argparse.ArgumentParser()
    # parse args
    parser.add_argument(
        "-type", default="default", type=str, help="model type, like vae or default"
    )
    parser.add_argument("-dataset", default="cifar", type=str)
    parser.add_argument("-epochs", default=10, type=int)
    if len(sys.argv) == 2:
        # model type is only command line arg
        model_type = sys.argv[1]
        dataset = "cifar"
        epochs = 10
    else:
        args = parser.parse_args()
        model_type = args.type
        dataset = args.dataset
        epochs = args.epochs
    train_encoder(model_type, dataset, epochs)


if __name__ == "__main__":
    main()
