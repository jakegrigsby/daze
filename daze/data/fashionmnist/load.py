import os
import zipfile

import pandas as pd
import numpy as np

import daze as dz

def load(size=None, dtype=None):
    if zipfile.is_zipfile('fashion-mnist_train.zip'):
        with zipfile.ZipFile('fashion-mnist_train.zip', 'r') as trainzip:
            trainzip.extractall()
        os.remove('fashion-mnist_train.zip')
    if zipfile.is_zipfile('fashion-mnist_test.zip'):
        with zipfile.ZipFile('fashion-mnist_test.zip', 'r') as testzip:
            testzip.extractall()
        os.remove('fashion-mnist_test.zip')
    x_train = pd.read_csv(os.path.abspath('fashion-mnist_train.csv')).values[:,1:]
    x_val = pd.read_csv(os.path.abspath('fashion-mnist_test.csv')).values[:,1:]
    if size:
        x_train = x_train[:size]
        x_val = x_val[:size]
    dtype = dz.data.utils.parse_dtype(dtype)
    x_train, x_val = x_train.astype(dtype), x_val.astype(dtype)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    return x_train, x_val

if __name__ == "__main__":
    load()
