
import numpy as np

from daze.data.utils import relative_path, parse_dtype

def load(size=None, dtype=None, return_labels=False):
    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    dataset = np.load(relative_path(__file__, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True, encoding='bytes')
    imgs = dataset['imgs']
    if dtype:
        dtype = parse_dtype(dtype)
        imgs = imgs.astype(dtype)
    latents_values = dataset['latents_values']
    latents_classes = dataset['latents_classes']
    metadata = dataset['metadata'][()]
    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],np.array([1,])))
    if size:
        imgs = imgs[:size]
        latents_values = latents_values[:size]
    if return_labels:
        return imgs, latents_values
    return imgs
