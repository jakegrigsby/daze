""" Samples from a probabalistic distribution and displays output images. """

import argparse
import matplotlib.pyplot as plt
import numpy as np

import daze as dz
from daze.nets.encoders import *
from daze.nets.decoders import *

def load_model_from_checkpoint_path(checkpoint_path, latent_dim=16):
    #  @TODO: save model class info to checkpoint so this, and latent_dim, is 
        # inferred automatically. For now, just support VAEs
    encoder = ConvolutionalEncoder(latent_dim=latent_dim)
    if 'mnist' in checkpoint_path:
        decoder = MnistDecoder()
        image_size = 28
    elif 'cifar' in checkpoint_path:
        decoder = CifarDecoder()
        image_size = 32
    else:
        raise ValueError('Cannot infer datatype from path')
    model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
    model.load_weights(checkpoint_path)
    return model, image_size, latent_dim


def display_sample(checkpoint_path, grid_size):
    print('** loading model')
    model, image_size, latent_dim = load_model_from_checkpoint_path(checkpoint_path)
    print('** loaded model')
    # sample from grid
    # @TODO walk through all dimensions not just 2
    nx = ny = grid_size
    meshgrid = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
    meshgrid = np.array(meshgrid).reshape(2, nx*ny).T
    extra_dim = np.tile(np.random.normal(size=(1,latent_dim-2)), (nx*ny,1))
    z = np.concatenate((meshgrid, extra_dim), axis=1)
    z = z.astype(np.float32, copy=False)
    x_grid = model.decode(z)
    num_channels = x_grid.shape[-1]
    x_grid = x_grid.numpy().reshape(nx, ny, image_size, image_size, num_channels)
    # fill canvas
    canvas = np.zeros((nx*image_size, ny*image_size, num_channels)).squeeze()
    for xi in range(nx):
        for yi in range(ny):
            if num_channels > 1:
                canvas[xi*image_size:(xi+1)*image_size, yi*image_size:(yi+1)*image_size,:] = x_grid[xi, yi,:,:,:].squeeze()
            else:
                canvas[xi*image_size:(xi+1)*image_size, yi*image_size:(yi+1)*image_size] = x_grid[xi, yi,:,:,:].squeeze()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(canvas, cmap=plt.cm.Greys)
    ax.axis('off')
    print('** showing plot')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to saved model')
    parser.add_argument('--grid_size', type=int,  default=10, 
        help='size of image grid')
    args = parser.parse_args()
    display_sample(args.checkpoint_path, args.grid_size)

if __name__ == "__main__":
    main()