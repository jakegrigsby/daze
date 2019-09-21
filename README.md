# DAZE
![](docs/source/media/logo.png)
## Autoencoders and GANs

Daze is a library for Autoencoders and Generative Adversarial Networks. It includes implementations of a number of popular algorithms - and is designed to make it easy to research and develop new techniques.

## Installation
```bash
git clone https://github.com/jakegrigsby/daze.git
cd daze
make user
```

## Getting Started
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
|Autoencoder| [Baldi](https://dl.acm.org/citation.cfm?id=3045801) |
|GAN| [Goodfellow et al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
|Contractive AE | [Rifai et al.](http://www.icml-2011.org/papers/455_icmlpaper.pdf)
|Denoising AE| [Vincent et al.](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
|Sparse AE| [Ng](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
|VAE| [Kingma, Welling](https://arxiv.org/abs/1312.6114)
|BetaVAE| [Higgins et al.](https://openreview.net/references/pdf?id=Sy2fzU9gl)
|InfoVAE|[Zhao et al.](https://arxiv.org/abs/1706.02262)
