from .core import Model
from . import preprocessing
from . import loss
from . import forward_pass


"""
def ContractiveAutoEncoder(encoder, decoder, gamma, preprocessing_steps=[]):
    loss_funcs = [loss.reconstruction(), loss.contractive(0.1)]
    return Model(encoder, decoder, loss_funcs=loss_funcs)
"""


def DenoisingAutoEncoder(encoder, decoder, gamma, preprocessing_steps=[]):
    preprocessing_steps += [preprocessing.random_mask(gamma)]
    loss_funcs = [loss.denoising_reconstruction()]
    return Model(
        encoder, decoder, preprocessing_steps=preprocessing_steps, loss_funcs=loss_funcs
    )


def VariationalAutoEncoder(encoder, decoder, preprocessing_steps=[]):
    loss_funcs = [loss.reconstruction(), loss.kl(1.0)]
    return Model(
        encoder,
        decoder,
        preprocessing_steps=preprocessing_steps,
        loss_funcs=loss_funcs,
        forward_pass_func=forward_pass.probabalistic_encode_decode,
    )


def BetaVariationalAutoEncoder(encoder, decoder, beta, preprocessing_steps=[]):
    loss_funcs = [loss.reconstruction(), loss.kl(beta)]
    return Model(
        encoder,
        decoder,
        preprocessing_steps=preprocessing_steps,
        forward_pass_func=forward_pass.probabalistic_encode_decode,
        loss_funcs=loss_funcs,
    )


def KlSparseAutoEncoder(encoder, decoder, rho, beta, preprocessing_steps=[]):
    loss_funcs = [loss.reconstruction(), loss.sparsity(rho, beta)]
    return Model(
        encoder, decoder, preprocessing_steps=preprocessing_steps, loss_funcs=loss_funcs
    )


def L1SparseAutoEncoder(encoder, decoder, gamma, preprocessing_steps=[]):
    loss_funcs = [loss.reconstruction(), loss.latent_l1(gamma)]
    return Model(
        encoder, decoder, preprocessing_steps=preprocessing_steps, loss_funcs=loss_funcs
    )
