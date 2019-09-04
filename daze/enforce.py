import functools

import daze as dz

class compatible:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
        self.func._compatability = []

    def add_compatability(self, code):
        self.func._compatability.append(code)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    @property
    def compatability(self):
        return self.func._compatability

class ae_compatible(compatible):
    def __init__(self, func):
        super().__init__(func)
        self.add_compatability("AE")

class vae_compatible(ae_compatible):
    def __init__(self, func):
        super().__init__(func)
        self.add_compatability("VAE")

class gan_compatible(compatible):
    def __init__(self, func):
        super().__init__(func)
        self.add_compatability("GAN")

class discriminator_compatible(gan_compatible):
    def __init__(self, func):
        super().__init__(func)
        self.add_compatability("DISC")

class generator_compatible(gan_compatible):
    def __init__(self, func):
        super().__init__(func)
        self.add_compatability("GEN")

