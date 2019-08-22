import os
import pathlib
import re

import numpy as np
import scipy.io.wavfile

from daze.data.utils import download, unzip_if_zipped, relative_path, parse_dtype


def load(size=None, dtype=None, return_labels=False):
    """Sample rate is 44100"""
    if not os.path.exists(relative_path(__file__, 'ESC-50-master')):
        download('https://github.com/karoldvl/ESC-50/archive/master.zip')
    unzip_if_zipped(relative_path(__file__, 'ESC-50-master'))
    wavs_root = pathlib.Path(relative_path(__file__, 'ESC-50-master/audio'))
    all_wav_paths = list(wavs_root.glob('*'))
    all_wav_paths = [str(path) for path in all_wav_paths]

    recordings = []
    labels = []
    label_regex = re.compile(r"(\d+).wav$")
    for filepath in all_wav_paths:
        _, audio =  scipy.io.wavfile.read(filepath)
        recordings.append(audio)
        label = int(label_regex.search(filepath)[1])
        labels.append(label)
    recordings = np.asarray(recordings)
    labels = np.asarray(labels)

    if dtype:
        dtype = parse_dtype(dtype)
        recordings = recordings.astype(dtype)
        labels = labels.astype(dtype)

    if size:
        recordings = recordings[:size]
        labels = labels[:size]

    if return_labels:
        return recordings, labels
    return recordings
    
