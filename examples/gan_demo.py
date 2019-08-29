import daze as dz

x_train_np, x_val_np = dz.data.mnist.load(dtype="f32")
x_train_np /= 255
x_val_np /= 255

noise_dim=100
sample_seed = dz.math.random_normal([5, noise_dim])

x_train = dz.data.utils.convert_np_to_tf(x_train_np, batch_size=32)
x_val = dz.data.utils.convert_np_to_tf(x_val_np, batch_size=32)

discriminator = dz.nets.encoders.ConvolutionalEncoder()
generator = dz.nets.decoders.MnistDecoder()

model = dz.model.GAN(generator, discriminator, noise_dim=noise_dim, discriminator_loss=[dz.loss.one_sided_label_smoothing()], generator_loss=[dz.loss.feature_matching()])

model.train(x_train, x_val, verbosity=2, save_path="saves/gan_demo",
            callbacks=[dz.callbacks.tensorboard_generative_sample(sample_seed)])
