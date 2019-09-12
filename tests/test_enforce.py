import pytest

import daze as dz
from daze.nets.encoders import ConvolutionalEncoder
from daze.nets.decoders import CifarDecoder
from daze.enforce import DazeModelTypeError

x_train_np, x_val_np = dz.data.cifar10.load(64, "f32")
sample_img = x_train_np[0]
x_train_np /= 255
x_val_np /= 255

x_train = dz.data.utils.convert_np_to_tf(x_train_np, batch_size=32)
x_val = dz.data.utils.convert_np_to_tf(x_val_np, batch_size=32)

def test_gan_with_vae_forward_pass():
    with pytest.raises(DazeModelTypeError):
        model = dz.GAN(CifarDecoder(), ConvolutionalEncoder(), 100, forward_pass_func=dz.forward_pass.probabilistic_encode_decode())

def test_gan_with_ae_loss_in_gen_loss():
    with pytest.raises(DazeModelTypeError):
        model = dz.GAN(CifarDecoder(), ConvolutionalEncoder(), 100, generator_loss=[dz.loss.contractive(.1)])

def test_gan_with_ae_loss_in_disc_loss():
    with pytest.raises(DazeModelTypeError):
        model = dz.GAN(CifarDecoder(), ConvolutionalEncoder(), 100, discriminator_loss=[dz.loss.reconstruction()])

def test_gan_with_disc_loss_in_gen_loss():
    with pytest.raises(DazeModelTypeError):
        model = dz.GAN(CifarDecoder(), ConvolutionalEncoder(), 100, generator_loss=[dz.loss.one_sided_label_smoothing()])

def test_gan_with_gen_loss_in_disc_loss():
    with pytest.raises(DazeModelTypeError):
        model = dz.GAN(CifarDecoder(), ConvolutionalEncoder(), 100, discriminator_loss=[dz.loss.vanilla_generator_loss()])

def test_ae_with_gan_forward_pass():
    with pytest.raises(DazeModelTypeError):
        model = dz.AutoEncoder(ConvolutionalEncoder(3), CifarDecoder(), forward_pass_func=dz.forward_pass.generative_adversarial())

def test_ae_with_vae_forward_pass():
    with pytest.raises(DazeModelTypeError):
        model = dz.AutoEncoder(ConvolutionalEncoder(3), CifarDecoder(), forward_pass_func=dz.forward_pass.probabilistic_encode_decode(), loss_funcs=[dz.loss.latent_l1()])

def test_ae_with_vae_loss_func():
    with pytest.raises(DazeModelTypeError):
        model = dz.AutoEncoder(ConvolutionalEncoder(3), CifarDecoder(), loss_funcs=[dz.loss.kl()])

def test_ae_with_gan_loss_func():
    with pytest.raises(DazeModelTypeError):
        model = dz.AutoEncoder(ConvolutionalEncoder(3), CifarDecoder(), loss_funcs=[dz.loss.feature_matching()])







