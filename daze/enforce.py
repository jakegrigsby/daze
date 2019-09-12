import functools

import daze as dz

class compatible:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class encoder_decoder_compatible(compatible):
    def __init__(self, func):
        super().__init__(func)

class ae_compatible(encoder_decoder_compatible):
    def __init__(self, func):
        super().__init__(func)

class vae_compatible(encoder_decoder_compatible):
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

class DazeModelTypeError(Exception):
    pass

def check_compatability(object_, type_):
    if isinstance(object_, type_):
        return
    elif type(object_) in type_.__bases__:
        return
    raise DazeModelTypeError

