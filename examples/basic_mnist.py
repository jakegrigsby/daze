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
