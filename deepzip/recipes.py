from .core import BaseModel
from .preprocessing import random_mask
from .loss import base_vae_loss

def DenoisingAutoEncoder():
    return BaseModel(preprocessing=random_mask(gamma))

def VariationalAutoEncoder():
    return BaseModel(loss=base_vae_loss)
