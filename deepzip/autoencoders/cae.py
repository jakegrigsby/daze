
import tensorflow as tf

import deepzip as dz

class ContractiveAutoEncoder(dz.core.BaseModel):
    def __init__(self, input_shape, encode_block, decode_block, gamma):
        super().__init__(input_shape, encode_block, decode_block)
        self.gamma = gamma

    @tf.function
    def compute_loss(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        contractive_term = dz.loss.code_frobenius_norm(h, x)
        loss = tf.losses.mean_squared_error(x, x_hat) + self.gamma * contractive_term


