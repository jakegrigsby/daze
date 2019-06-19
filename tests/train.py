import deepzip as dz
from deepzip.autoencoders.encoders import EasyEncoder
from deepzip.autoencoders.decoders import EasyDecoder

import argparse
import tensorflow as tf

def train_encoder(encoder_type):
    if encoder_type == 'default':
        model = dz.core.AutoEncoder(EasyEncoder, EasyDecoder)
    elif encoder_type == 'vae':
        model = dz.recipes.VariationalAutoEncoder(EasyEncoder, EasyDecoder)
    elif encoder_type == 'denoising':
        model = dz.recipes.DenoisingAutoEncoder(EasyEncoder, EasyDecoder, .15)
    else:
        raise ValueError('Invalid autoencoder type {}'.format(encoder_type))

    x_train, x_val = dz.data.cifar10.load()
    print('Model:', encoder_type)
    model.train(x_train, x_val, epochs=10, experiment_name='vae_test')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='model type, like vae or default')
    args = parser.parse_args()
    train_encoder(args.type)

main()
