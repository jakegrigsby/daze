import os
import pathlib

import numpy as np
import tensorflow as tf

import daze as dz
from daze.data.utils import relative_path, download, untar_if_tarred, parse_dtype

def load(size=None, dtype=None):
    if not os.path.exists(relative_path(__file__, 'rendered_chairs.tar')) and \
            not os.path.exists(relative_path(__file__, 'rendered_chairs')):
        print("Downloading chairs dataset... (this could take a while)")
        download('http://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar')
    untar_if_tarred(relative_path(__file__,'rendered_chairs.tar'))

    imgs_root = pathlib.Path(relative_path(__file__,'rendered_chairs'))
    all_image_paths = list(imgs_root.glob('*/*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
     
    dtype = parse_dtype(dtype)
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        if dtype:
            image = tf.dtypes.cast(image, dtype)
        return image

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if size:
        image_ds = image_ds.take(size)
    return image_ds
