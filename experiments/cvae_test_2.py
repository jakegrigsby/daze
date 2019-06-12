import deepzip as dz
import tensorflow as tf

def load_mnist():
	(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

	# Normalizing the images to the range of [0., 1.]
	train_images /= 255.
	test_images /= 255.

	# Binarization
	train_images[train_images >= .5] = 1.
	train_images[train_images < .5] = 0.
	test_images[test_images >= .5] = 1.
	test_images[test_images < .5] = 0.
	TRAIN_BUF = 60000
	BATCH_SIZE = 100

	TEST_BUF = 10000

	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
	test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

	return train_dataset, test_dataset

x_train, x_val = load_mnist()
model = dz.autoencoders.CVAE(input_shape=(28,28,1), latent_dim=50)
model.summary()
model.train(x_train, x_val, 10, 'conv_ae')
