
import daze as dz

x_train_np, x_val_np = dz.data.mnist.load(dtype="f32")
x_train_np /= 255
x_val_np /= 255

x_train = dz.data.utils.convert_np_to_tf(x_train_np, batch_size=32)
x_val = dz.data.utils.convert_np_to_tf(x_val_np, batch_size=32)

discriminator = dz.nets.encoders.ConvolutionalEncoder(latent_dim=1)
generator = dz.nets.decoders.MnistDecoder()

model = dz.model.GAN(generator, noise_dim=100, discriminator=discriminator)

model.train(x_train, x_val, verbosity=2)

