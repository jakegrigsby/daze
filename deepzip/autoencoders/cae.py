
import deepzip as dz

class ContractiveAutoEncoder(dz.core.BaseModel):
    def __init__(self, encoder_block, decoder_block, gamma):
        super().__init__(encode_block, decode_block)

    def compute_loss(self, x):
        h = self.encode(x)
        dh_dx = tf.expand_dims(tf.convert_to_tensor(tf.gradients(h, x), dtype=tf.float32),0)
        frob_norm = tf.norm(dh_dx, ord='fro')
        x_hat = self.decode(h)
        loss = tf.losses.mean_squared_error(x, x_hat) + gamma * frob_norm


