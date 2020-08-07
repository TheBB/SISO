import numpy as np


def prod(values):
    retval = 1
    for value in values:
        retval *= value
    return retval


def flatten_2d(array):
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array.reshape((-1, array.shape[-1]))


def ensure_ncomps(data, ncomps: int, allow_scalar: bool):
    assert data.ndim == 2
    if data.shape[-1] == 1 and allow_scalar:
        return data
    if data.shape[-1] >= ncomps:
        return data
    return np.hstack([data, np.zeros((data.shape[0], ncomps - data.shape[-1]), dtype=data.dtype)])
