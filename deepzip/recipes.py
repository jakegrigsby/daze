from .core import Model
from . import preprocessing
from . import loss
from . import forward_pass

def ContractiveAutoEncoder(encoder, decoder, gamma):
    return Model(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.contractive(.1)])

def DenoisingAutoEncoder(encoder, decoder, gamma):
    return Model(encoder, decoder, preprocessing_steps=[preprocessing.random_mask(gamma)], loss_funcs=[loss.denoising_reconstruction()])

def VariationalAutoEncoder(encoder, decoder):
    return Model(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.kl(1.)], forward_pass=forward_pass.probabalistic_encode_decode)

def BetaVariationalAutoEncoder(encoder, decoder, beta):
    return Model(encoder, decoder, forward_pass=forward_pass.probabalistic_encode_decode, loss_funcs=[loss.reconstruction(), loss.kl(beta)])

def KlSparseAutoEncoder(encoder, decoder, rho, beta):
    return Model(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.sparsity(rho, beta)])

def L1SparseAutoEncoder(encoder, decoder, gamma):
    return Model(encoder, decoder, loss_funcs=[loss.reconstruction(), loss.latent_l1(gamma)])
