# DAZE
![](docs/source/media/logo.png)
## Autoencoders

TODO: intro here...


Daze includes a number of popular datasets and algorithms out-of-the-box. Here's a short script that trains a Denoising Autoencoder on MNIST.
```python
import daze as dz
from daze.callbacks import *
from daze.data.utils import convert_np_to_tf

train, val = dz.data.mnist.load(dtype="f32")
train, val = train/255., val/255.

encoder = dz.nets.encoders.ConvolutionalEncoder(latent_dim=3)
decoder = dz.nets.decoders.MnistDecoder()
model = dz.recipes.DenoisingAutoEncoder(encoder, decoder)

callbacks = [checkpoints(1), tensorboard_loss_scalars(), tensorboard_latent_space_plot(train[:100]), tensorboard_image_reconstruction(train[:5])]

train_tf, val_tf = convert_np_to_tf(train, batch_size=32), convert_np_to_tf(val, batch_size=32)
model.train(train_tf, val_tf, epochs=20, verbosity=2, callbacks=callbacks)
```
This script runs 20 epochs of training, saving weights along the way. It also generates diagrams like these - showing the quality of the reconstruction at the end of each epoch.

![](docs/source/media/reconstructions_mnist.png)

Because the latent dimension size was <= 3, the `tensorboard_latent_space_plot` callback drew this diagram:

![](docs/source/media/latent_space_mnist.png)

The `recipes` module includes implementations of:

| Name | Paper |
| --- | :---: | 
|Contractive AE | [Rifai et al.](http://Jakes-MacBook-Pro-2.local:6006/)
|Denoising AE| []
|Sparse AE|
|VAE|
|BetaVAE|
|InfoVAE|

