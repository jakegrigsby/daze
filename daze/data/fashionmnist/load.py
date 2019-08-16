
import pandas as pd
import numpy as np

import daze as dz
from daze.data.utils import relative_path, unzip_if_zipped

def load(size=None, dtype=None):
    train_zip_path = relative_path(__file__, 'fashion-mnist_train.zip')
    unzip_if_zipped(train_zip_path)

    test_zip_path = relative_path(__file__, 'fashion-mnist_test.zip')
    unzip_if_zipped(test_zip_path)

    x_train = pd.read_csv(relative_path(__file__, 'fashion-mnist_train.csv')).values[:,1:]
    x_val = pd.read_csv(relative_path(__file__, 'fashion-mnist_test.csv')).values[:,1:]
    if size:
        x_train = x_train[:size]
        x_val = x_val[:size]
    dtype = dz.data.utils.parse_dtype(dtype)
    x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    return x_train, x_val
