
import pandas as pd
import numpy as np

import daze as dz
from daze.data.utils import relative_path, unzip_if_zipped, parse_dtype

def load(size=None, dtype=None, return_labels=False):
    train_zip_path = relative_path(__file__, 'fashion-mnist_train.zip')
    unzip_if_zipped(train_zip_path)

    test_zip_path = relative_path(__file__, 'fashion-mnist_test.zip')
    unzip_if_zipped(test_zip_path)

    train = pd.read_csv(relative_path(__file__, 'fashion-mnist_train.csv')).values
    val = pd.read_csv(relative_path(__file__, 'fashion-mnist_test.csv')).values
    x_train, y_train = train[:,1:], train[:,0]
    x_val, y_val = val[:,1:], val[:,0]
    if size:
        x_train = x_train[:size]
        x_val = x_val[:size]
        y_train = y_train[:size]
        y_val = y_val[:size]
    if dtype:
        dtype = parse_dtype(dtype)
        x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    if return_labels:
        return (x_train, y_train), (x_val, y_val)
    else:
        return x_train, x_val
