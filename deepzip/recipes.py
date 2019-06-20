from .core import AutoEncoder
from . import preprocessing
from . import loss

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.contractive(.1))

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)], loss=loss.reconstruction_sparsity(.1))

def VariationalAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.vae())

def SparseAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.reconstruction_sparsity(.1))
