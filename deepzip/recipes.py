from .core import AutoEncoder
from .preprocessing import random_mask
from .loss import compute_loss_vae

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(*args, **kwargs):
    return NotImplementedError('Fuck you jake')

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return AutoEncoder(encoder, decoder, preprocessing=random_mask(gamma))

def VariationalAutoEncoder(encoder, decoder):
    return AutoEncoder(encoder, decoder, loss=compute_loss_vae)
