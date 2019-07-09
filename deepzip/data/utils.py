import numpy as np


def get_byte_count(s):
    idx = -1
    if not s[idx].isnumeric():
        return None
    while s[idx - 1 :].isnumeric():
        idx -= 1
    return int(s[idx:])


def parse_dtype(s):
    if not type(s) == str:
        # s is already in correct numpy dtype form
        return s
    err = lambda: ValueError(f"Datatype string code not recognized: {s}")
    byte_count = get_byte_count(s)
    if s[0] == "f":
        # float
        if not byte_count:
            return np.float32
        elif byte_count in (16, 32, 64):
            return eval(f"np.float{byte_count}")
        else:
            return err()
    elif s[0] == "i":
        # int
        if not byte_count:
            return np.int32
        elif byte_count in (8, 16, 32, 64):
            return eval(f"np.int{byte_count}")
        else:
            return err()
    elif s[0] == "u":
        # unsigned int
        if not byte_count:
            return np.uint32
        elif byte_count in (8, 16, 32, 64):
            return eval(f"np.uint{byte_count}")
    else:
        return err()
