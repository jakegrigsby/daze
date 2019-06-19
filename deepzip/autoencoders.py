from .core import BaseModel
from .preprocessing import random_mask
from .loss import compute_loss_vae

#@TODO make this pattern into a decorator

def ContractiveAutoEncoder(*args, **kwargs):
    return NotImplementedError('Fuck you jake')

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return BaseModel(encoder, decoder, preprocessing=random_mask(gamma))

def VariationalAutoEncoder():
    return BaseModel(encoder, decoder, loss=compute_loss_vae)
