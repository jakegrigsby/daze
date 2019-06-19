from .core import BaseModel
from .preprocessing import random_mask
from .loss import compute_loss_vae

#@TODO make this pattern into a decorator

def DenoisingAutoEncoder(gamma, *args, **kwargs):
    return BaseModel(*args, preprocessing=random_mask(gamma), **kwargs)

def VariationalAutoEncoder():
    return BaseModel(*args, loss=compute_loss_vae, **kwargs)
