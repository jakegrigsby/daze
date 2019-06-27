from .core import AutoEncoder, GenerativeAutoEncoder
from . import preprocessing
from . import loss

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.contractive(.1)])

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)], loss_funcs=[loss.denoising()])

def VariationalAutoEncoder(encoder, decoder, latent_dim=None):
    return GenerativeAutoEncoder(encoder, decoder, call_func='vae', loss_funcs=[loss.vae()], latent_dim=latent_dim)

def SparseAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.sparsity(gamma)])
