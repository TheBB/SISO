from functools import reduce
from itertools import chain, product, count
from pathlib import Path
import logging

import numpy as np

from typing import (
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)


def pluralize(num: int, singular: str, plural: str) -> str:
    return f'{num} {singular if num == 1 else plural}'


def flatten_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array.reshape(-1, array.shape[-1])


def transpose_butlast(array: np.ndarray) -> np.ndarray:
    last = array.ndim - 1
    permutation = tuple(reversed(range(last))) + (last,)
    return array.transpose(permutation)


T = TypeVar('T')

def _pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    it = iter(iterable)
    left = next(it)
    for right in it:
        yield left, right
        left = right


def subdivide_linear(knots: np.ndarray, nvis: int) -> np.ndarray:
    return np.fromiter(chain(
        chain.from_iterable(
            (((nvis - i) * a + i * b) / nvis for i in range(nvis))
            for a, b in _pairwise(knots)
        ),
        (knots[-1],),
    ), float)


def prod(values: Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, values, 1)


def first_and_has_more(values: Iterable[T]) -> Tuple[T, bool]:
    it = iter(values)
    first = next(it)
    try:
        next(it)
        return first, True
    except StopIteration:
        return first, False


def structured_cells(cellshape: Tuple[int, ...], pardim: int, nodemap: Optional[np.ndarray]=None):
    nodeshape = tuple(s + 1 for s in cellshape)
    ranges = [range(k) for k in cellshape]
    nidxs = [np.array(q) for q in zip(*product(*ranges))]
    eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)), dtype=int)
    if pardim == 1:
        eidxs[:,0] = nidxs[0]
        eidxs[:,1] = nidxs[0] + 1
    elif pardim == 2:
        i, j = nidxs
        eidxs[:,0] = np.ravel_multi_index((i, j), nodeshape)
        eidxs[:,1] = np.ravel_multi_index((i+1, j), nodeshape)
        eidxs[:,2] = np.ravel_multi_index((i+1, j+1), nodeshape)
        eidxs[:,3] = np.ravel_multi_index((i, j+1), nodeshape)
    elif pardim == 3:
        i, j, k = nidxs
        eidxs[:,0] = np.ravel_multi_index((i, j, k), nodeshape)
        eidxs[:,1] = np.ravel_multi_index((i+1, j, k), nodeshape)
        eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nodeshape)
        eidxs[:,3] = np.ravel_multi_index((i, j+1, k), nodeshape)
        eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nodeshape)
        eidxs[:,5] = np.ravel_multi_index((i+1, j, k+1), nodeshape)
        eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nodeshape)
        eidxs[:,7] = np.ravel_multi_index((i, j+1, k+1), nodeshape)

    if nodemap is not None:
        eidxs = nodemap.flat[eidxs]

    return eidxs


def filename_generator(basename: Path, instantaneous: bool) -> Iterator[Path]:
    if instantaneous:
        yield basename
        return
    stem = basename.stem
    suffix = basename.suffix
    for i in count(1):
        yield basename.with_name(f'{stem}-{i}').with_suffix(suffix)
