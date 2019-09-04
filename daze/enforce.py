import functools

import daze as dz

class compatible:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ae_compatible(compatible):
    def __init__(self, func):
        super().__init__(func)

class vae_compatible(ae_compatible):
    def __init__(self, func):
        super().__init__(func)

class gan_compatible(compatible):
    def __init__(self, func):
        super().__init__(func)

class discriminator_compatible(gan_compatible):
    def __init__(self, func):
        super().__init__(func)

class generator_compatible(gan_compatible):
    def __init__(self, func):
        super().__init__(func)
