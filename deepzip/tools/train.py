import argparse
import sys

import tensorflow as tf

import deepzip as dz
from deepzip.nets.encoders import *
from deepzip.nets.decoders import *
from deepzip.callbacks import checkpoints, tensorboard
from deepzip.preprocessing import basic_image_normalize


def train_encoder(model_type, dataset_type, epochs, batch_size=None):
    # Use dataset to infer encoder, decoder
    if dataset_type in ["cifar", "cifar10"]:
        dataset = dz.data.cifar10
        encoder = Encoder_32x32()
        decoder = Decoder_32x32()
    elif dataset_type in ["mnist"]:
        dataset = dz.data.mnist
        encoder = EasyEncoder()
        decoder = EasyDecoder()
    else:
        raise ValueError(f"Invalid dataset code {datset_type}. Options are: 'cifar'...")

    if batch_size:
        x_train, x_val = dataset.load(batch_size, dtype="f")
    else:
        x_train, x_val = dataset.load(dtype="f")
        
    x_train /= 255
    x_train = dz.data.utils.convert_np_to_tf(x_train, 32)
    x_val /= 255
    x_val = dz.data.utils.convert_np_to_tf(x_val, 32)
    callbacks = [checkpoints(1), tensorboard()]

    # Select algorithm type
    if model_type == "default":
        model = dz.Model(encoder, decoder)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/default",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "vae":
        model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/vae",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "denoising":
        model = dz.recipes.DenoisingAutoEncoder(encoder, decoder, gamma=0.1)
        
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/denoising",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "l1sparse":
        model = dz.recipes.L1SparseAutoEncoder(encoder, decoder, gamma=0.1)
        
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path="saves/l1sparse",
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "contractive":
        model = dz.recipes.ContractiveAutoEncoder(encoder, decoder, gamma=0.1)
        
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
            encoder, decoder, beta=0.1
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
            encoder, decoder, rho=0.01, beta=0.1
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
    parser.add_argument("--batch_size", default=None, type=int)
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
    train_encoder(model_type, dataset, epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
