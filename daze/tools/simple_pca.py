import argparse
import time

from matplotlib.widgets import Button
from sklearn.decomposition import PCA

import daze as dz
from daze.nets.encoders import *
from daze.nets.decoders import *
from daze.callbacks import *

# Number of different images to draw in step.
NUM_DRAW_STEPS = 25
# Amount of seconds to wait between displays during Draw phase.
DRAW_WAIT_TIME = 0.03

# Global data for Draw. @TODO abstract to a class and bind these.
DRAW_STARTED = False
DRAW_START_COORDS = None
DRAW_START_POINT = None
DRAW_END_POINT = None

def display(model, data, num_samples):
    data_x, data_y = data
    image_shape = data_x.squeeze().shape[1:]
    
    if len(image_shape) == 2:
        cmap = 'Greys'
    else:
        cmap = None
    samples_x = data_x[:num_samples, :, :, :]
    
    samples_y = data_y.squeeze()[:num_samples]
    
    encoded_x = model.encode(samples_x)
    
    pca = PCA(n_components=2)
    pca.fit(encoded_x)
    pca_x = pca.transform(encoded_x)
    
    """ Show scatterplot. """
    colors=['red','blue','green','orange','purple','cyan','brown','yellow','pink', 'gray']
    target_names=['zero','one','two','three','four','five','six','seven','eight','nine']
    line_width=2
    
    fig_canvas = plt.figure().canvas
    
    left_subplot =  plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    
    for color, i, target_name in zip(colors, list(range(10)), target_names):
        left_subplot.scatter(pca_x[samples_y == i, 0], pca_x[samples_y == i, 1], 
            color=color, alpha=.8, lw=line_width, label=target_name)
    
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset')
    
    """ Method for sampling from grid and redrawing a new image. """
    
    def sample_from_pca(list_of_coords, sleep_time=None):
        latent_representation = pca.inverse_transform(np.array(list_of_coords))
        latent_representation = latent_representation.astype('float32')
        mean, sigma = np.split(latent_representation, 2, axis=1)
        all_image_data = model.decode(mean).numpy()
        for i in range(len(list_of_coords)):
            image_data = all_image_data[i].reshape(image_shape)
            coords_before_pca = list_of_coords[i]
            if DRAW_START_POINT:
                DRAW_START_POINT.set_offsets(coords_before_pca)
            image_sidebox.set_data(image_data)
            if sleep_time: 
                # Use plt.pause over time.sleep to save yourself a massive headache.
                plt.pause(sleep_time)
            plt.draw()
    
    """ Onclick method. """
    
    def on_plot_clicked(event):
        global DRAW_STARTED, DRAW_START_COORDS, DRAW_START_POINT, DRAW_END_POINT
        
        if not event.inaxes or event.inaxes == button_axes:
            # Must click within scatterplot
            return
        ix, iy = event.xdata, event.ydata
        if DRAW_STARTED:
            # Get coordinates for original click
            # @TODO: display dot on screen for this drawing part
            if not DRAW_START_COORDS:
                sample_from_pca([[ix, iy]], sleep_time=None)
                DRAW_START_POINT = left_subplot.scatter([ix], [iy], marker='2', color='black', alpha=1, lw=1)
                DRAW_START_COORDS = (ix, iy)
            else:
                DRAW_END_POINT = left_subplot.scatter([ix], [iy], marker='s', color='black', alpha=1, lw=1)
                
                all_coords = np.linspace(DRAW_START_COORDS, (ix, iy), num=NUM_DRAW_STEPS)
                sample_from_pca(all_coords, sleep_time=DRAW_WAIT_TIME)
                
                DRAW_START_COORDS = None
                DRAW_STARTED = False
                
                DRAW_START_POINT.remove()
                DRAW_START_POINT = None
                
                DRAW_END_POINT.remove()
                DRAW_END_POINT = None
                
                button.set_active(True)
        else:
            print('')
            # if Draw disabled, just sample from grid normally
            sample_from_pca([[ix, iy]], sleep_time=None)
    
    fig_canvas.mpl_connect('button_press_event', on_plot_clicked)
    
    """ Show image on sidebar. """
    right_subplot = plt.subplot2grid((3,3), (0, 2), colspan=3)
    random_row = np.random.choice(len(samples_x))
    random_sample = np.array([samples_x[random_row]], dtype='float32')
    original_image = model.call(None, random_sample)['x_hat'].numpy().reshape(image_shape)
                # @TODO model should have easy way to do this, like a `reconstruct` method
                
    image_padding = 5 # percent
    image_extent = tuple([image_padding, 100 - image_padding] * 2)
    image_sidebox = right_subplot.imshow(original_image, 
                      extent=image_extent,
                      cmap=cmap) #@TODO: fix this to center image properly
    
    """ Button for drawing trajectories. """
    def on_button_click(event):
        global DRAW_STARTED
        DRAW_STARTED = True
        button.set_active(False)
    
    button_axes = plt.axes([0.8, 0.05, 0.1, 0.1]) # @TODO: Optimize these margins
    button = Button(button_axes, 'Draw', color='#E0E0E0', hovercolor='Gray')
    button.on_clicked(on_button_click)
    
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
