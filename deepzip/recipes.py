from .core import AutoEncoder
from . import preprocessing
from . import loss

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.code_frobenius_norm)

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)])

def VariationalAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.compute_loss_vae)

def SparseAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=loss.reconstruction_sparsity)
