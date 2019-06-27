from .core import AutoEncoder, GenerativeAutoEncoder
from . import preprocessing
from . import loss

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss=loss.contractive(gamma))

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)], loss=loss.denoising())

def VariationalAutoEncoder(encoder, decoder, latent_dim=None):
    return GenerativeAutoEncoder(encoder, decoder, loss=loss.vae(), latent_dim=latent_dim)

def SparseAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss=loss.reconstruction_sparsity(gamma))
