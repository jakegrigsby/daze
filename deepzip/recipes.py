from .core import AutoEncoder
from . import preprocessing
from . import loss

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss=loss.contractive(gamma))

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)], loss=loss.denoising())

def VariationalAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.vae())

def SparseAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss=loss.reconstruction_sparsity(gamma))
