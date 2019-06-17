

import numpy as np
import tensorflow as tf

import deepzip as dz

class DenoisingAutoEncoder(dz.core.BaseModel):
    def process_input(self, inputs):
        return dz.noise.random_mask(inputs, .25)