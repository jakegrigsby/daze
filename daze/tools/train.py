import argparse
import os

import tensorflow as tf

import daze as dz
from daze.nets.encoders import *
from daze.nets.decoders import *
from daze.callbacks import checkpoints, tensorboard


def train_encoder(
    model_type, encoder, decoder, dataset, latent_size, epochs, save_path
):
    x_train, x_val = dataset
    callbacks = [checkpoints(1), tensorboard()]

    # Select algorithm type
    if model_type == "default":
        model = dz.Model(encoder, decoder)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "vae":
        model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "denoising":
        model = dz.recipes.DenoisingAutoEncoder(encoder, decoder, gamma=0.1)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "l1sparse":
        model = dz.recipes.L1SparseAutoEncoder(encoder, decoder, gamma=0.1)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "contractive":
        model = dz.recipes.ContractiveAutoEncoder(encoder, decoder, gamma=0.1)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "bvae":
        model = dz.recipes.BetaVariationalAutoEncoder(encoder, decoder, beta=1.1)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
            epochs=epochs,
            verbosity=2,
        )
    elif model_type == "klsparse":
        model = dz.recipes.KlSparseAutoEncoder(encoder, decoder, rho=0.01, beta=0.1)
        model.train(
            x_train,
            x_val,
            callbacks=callbacks,
            save_path=save_path,
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
    parser.add_argument("-latent_size", default=32, type=int)
    parser.add_argument("-save_path", type=str)
    parser.add_argument("-encoder", default="ConvolutionalEncoder")
    parser.add_argument("-decoder")
    args = parser.parse_args()
    model_type = args.type
    dataset_type = args.dataset
    epochs = args.epochs
    latent_size = args.latent_size

    # Use dataset to infer encoder, decoder
    encoder = ConvolutionalEncoder(latent_dim=latent_size)
    if dataset_type in ["cifar", "cifar10"]:
        x_train, x_val = dz.data.cifar10.load(dtype="f")
        x_train /= 255
        x_val /= 255
        decoder = CifarDecoder()
    elif dataset_type in ["mnist"]:
        x_train, x_val = dz.data.mnist.load(dtype="f")
        x_train /= 255
        x_val /= 255
        decoder = MnistDecoder()
    elif os.path.exists(dataset_type):
        dataset = dz.data.utils.load_from_file(dataset_type)
        x_train, x_val = dz.data.utils.train_val_split(dataset)
        x_train = x_train[0]
        x_val = x_val[0]
    else:
        raise ValueError(
            f"Dataset not recognized: {dataset_type}. Options are 'cifar', 'mnist' or a filepath to a csv."
        )

    # override default encoder, decoder
    if args.encoder:
        try:
            encoder = eval(args.encoder)
        except:
            raise ValueError(f"Encoder type not found: {args.encoder}")
        else:
            encoder = encoder(latent_dim=latent_size)

    if args.decoder:
        try:
            decoder = eval(args.decoder)
        except:
            raise ValueError(f"Decoder type not found: {args.decoder}")
        else:
            decoder = decoder()

    x_train = dz.data.utils.convert_np_to_tf(x_train, 32)
    x_val = dz.data.utils.convert_np_to_tf(x_val, 32)
    dataset = (x_train, x_val)

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = f"saves/{args.dataset}_{args.type}"

    train_encoder(model_type, encoder, decoder, dataset, latent_size, epochs, save_path)


if __name__ == "__main__":
    main()
