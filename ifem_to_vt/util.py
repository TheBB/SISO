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
