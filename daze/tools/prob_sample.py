""" Samples from a probabilistic distribution and displays output images. """

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def gaussian(x):
    mu = 0
    sig = 1
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def display_sample(checkpoint_path, num_steps, display_type='grid'):
    model, image_size, latent_dim = load_model_from_checkpoint_path(checkpoint_path)
    # sample from grid
    # @TODO walk through all dimensions not just 2
    
    sample_points = None
    for x_step in np.linspace(-3, 3, num=num_steps):
        y = gaussian(x_step)
        y_arr = np.array([y] * latent_dim, dtype='float32')
        if sample_points is None:
            sample_points = np.array([y_arr])
        else:
            sample_points = np.vstack((sample_points, y_arr))
        
    decoded_images = model.decode(sample_points)
    # show gif
    if display_type == 'gif':
        images = []
        fig = plt.figure()
        for raw_image in decoded_images:
            image = plt.imshow(np.squeeze(raw_image), animated=True)
            images.append([image])
        
        print('{} images'.format(len(images)))
        ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                        repeat_delay=1000)
    elif display_type == 'grid':
        # show grid
        nx=ny=int(np.sqrt(num_steps))
        f, axarr = plt.subplots(nx,ny)
        for _y in range(ny):
            for _x in range(nx):
                i = _x + (_y * nx)
                axarr[_x,_y].imshow(np.squeeze(decoded_images[i]))
    
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to saved model')
    parser.add_argument('--num_steps', type=int,  default=64, 
        help='number of steps across distribution')
    args = parser.parse_args()
    display_sample(args.checkpoint_path, args.num_steps)

if __name__ == "__main__":
    main()