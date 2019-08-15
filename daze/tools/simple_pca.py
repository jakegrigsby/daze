import argparse
import os

import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.decomposition import PCA

import daze as dz
from daze.nets.encoders import *
from daze.nets.decoders import *
from daze.callbacks import *

def display(model, data, num_samples):
    data_x, data_y = data
    image_shape = data_x.shape[1:]
    try:
        samples_x = data_x[:num_samples, :, :, :]
    except IndexError: # black and white
        samples_x = data_x[:num_samples, :, :]
        samples_x = np.expand_dims(samples_x, axis=-1)
    
    try:
        samples_y = data_y[:num_samples, :]
        samples_y = np.squeeze(samples_y)
    except IndexError:
        samples_y = data_y[:num_samples]
    print('*samples_x:', samples_x.shape)
    
    num_channels = samples_x.shape[-1]
    encoded_x = model.encode(samples_x)
    
    pca = PCA(n_components=2)
    pca.fit(encoded_x)
    pca_x = pca.transform(encoded_x)
    
    """ Show scatterplot. """
    colors=['red','blue','green','orange','purple','cyan','brown','yellow','black', 'gray']
    target_names=['zero','one','two','three','four','five','six','seven','eight','nine']
    line_width=2
    
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
    subplot1 =  plt.subplot(gs[0])
    
    for color, i, target_name in zip(colors, list(range(10)), target_names):
        subplot1.scatter(pca_x[samples_y == i, 0], pca_x[samples_y == i, 1], 
            color=color, alpha=.8, lw=line_width, label=target_name)
    
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset')
    
    subplot1_canvas = fig.canvas
    
    """ Show image on sidebar. """
    subplot2 = plt.subplot(gs[1])
    blank_image = np.zeros(image_shape, dtype='float32')
    image_sidebox = subplot2.imshow(blank_image)
    
    """ Onclick method. """
    
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        latent_representation = pca.inverse_transform(np.array([ix, iy]))
        latent_representation = np.expand_dims(latent_representation, axis=0)
        latent_representation = latent_representation.astype('float32')
        mean, sigma = np.split(latent_representation, 2, axis=1)
        image_data = model.decode(mean).numpy().reshape(image_shape)
        image_sidebox.set_data(image_data)
        plt.draw()
    
    cid = subplot1_canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()

def load_model_from_checkpoint_path(checkpoint_path, latent_dim):
    #  @TODO: save model class info to checkpoint so this, and latent_dim, is 
        # inferred automatically. For now, just support VAEs
    encoder = ConvolutionalEncoder(latent_dim=latent_dim)
    if 'mnist' in checkpoint_path:
        decoder = MnistDecoder()
    elif 'cifar' in checkpoint_path:
        decoder = CifarDecoder()
    else:
        raise ValueError('Cannot infer datatype from path')
    model = dz.recipes.VariationalAutoEncoder(encoder, decoder)
    model.load_weights(checkpoint_path)
    return model    

def load_dataset_from_path(dataset_path):
    if 'cifar' in dataset_path:
        (x_train, y_train), _ = dz.data.cifar10.load(dtype="f", return_labels=True)
        x_train /= 255
        decoder = CifarDecoder()
    elif "mnist" in dataset_path:
        (x_train, y_train), _ = dz.data.mnist.load(dtype="f", return_labels=True)
        x_train /= 255
        decoder = MnistDecoder()
    else:
        raise ValueError(
            f"Dataset not recognized: {dataset_path}. Options are 'cifar' or 'mnist'"
        )
    
    return x_train, y_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to saved model')
    parser.add_argument('--num_samples', '--N', type=int, default=100)
    parser.add_argument('--latent_dim', '--L', type=int, default=32)
    args = parser.parse_args()
    
    dataset = load_dataset_from_path(args.checkpoint_path)
    model = load_model_from_checkpoint_path(args.checkpoint_path, args.latent_dim)
    
    display(model, dataset, args.num_samples)

if __name__ == "__main__":
    main()
