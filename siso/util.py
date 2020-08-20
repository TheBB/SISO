from contextlib import contextmanager
from itertools import product

import numpy as np

from typing import Iterable, Type, TypeVar


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


T = TypeVar('T')
def subclasses(cls: Type[T], root: bool = False, invert: bool = False) -> Iterable[Type[T]]:
    """Iterate over all subclasses of CLS.  If ROOT is true, CLS itself is
    included.  If INVERT is true, yield subclasses before superclasses.
    """
    if root and not invert:
        yield cls
    for sub in cls.__subclasses__():
        if not invert:
            yield sub
        yield from subclasses(sub, root=False, invert=invert)
        if invert:
            yield sub
    if root and invert:
        yield cls


def subdivide_linear(knots, nvis):
    out = []
    for left, right in zip(knots[:-1], knots[1:]):
        out.extend(np.linspace(left, right, num=nvis, endpoint=False))
    out.append(knots[-1])
    return out


def subdivide_face(el, nodes, elements, nvis):
    left, bottom = el.start()
    right, top = el.end()
    xs = subdivide_linear((left, right), nvis)
    ys = subdivide_linear((bottom, top), nvis)

    for (l, r) in zip(xs[:-1], xs[1:]):
        for (b, t) in zip(ys[:-1], ys[1:]):
            sw, se, nw, ne = (l, b), (r, b), (l, t), (r, t)
            for pt in (sw, se, nw, ne):
                nodes.setdefault(pt, len(nodes))
            elements.append([nodes[sw], nodes[se], nodes[ne], nodes[nw]])


def subdivide_volume(el, nodes, elements, nvis):
    umin, vmin, wmin = el.start()
    umax, vmax, wmax = el.end()
    us = subdivide_linear((umin, umax), nvis)
    vs = subdivide_linear((vmin, vmax), nvis)
    ws = subdivide_linear((wmin, wmax), nvis)

    for (ul, ur) in zip(us[:-1], us[1:]):
        for (vl, vr) in zip(vs[:-1], vs[1:]):
            for (wl, wr) in zip(ws[:-1], ws[1:]):
                bsw, bse, bnw, bne = (ul, vl, wl), (ur, vl, wl), (ul, vr, wl), (ur, vr, wl)
                tsw, tse, tnw, tne = (ul, vl, wr), (ur, vl, wr), (ul, vr, wr), (ur, vr, wr)
                for pt in (bsw, bse, bnw, bne, tsw, tse, tnw, tne):
                    nodes.setdefault(pt, len(nodes))
                elements.append([nodes[bsw], nodes[bse], nodes[bne], nodes[bnw],
                                 nodes[tsw], nodes[tse], nodes[tne], nodes[tnw]])


def single_slice(total, axis, *args):
    index = [slice(None, None)] * total
    index[axis] = slice(*args)
    return tuple(index)


def single_index(total, axis, ix):
    index = [slice(None, None)] * total
    index[axis] = ix
    return tuple(index)


def unstagger(data, axis):
    index = [slice(None, None),] * data.ndim

    plus = list(index)
    plus[axis] = slice(1, None)
    plus = tuple(plus)

    minus = list(index)
    minus[axis] = slice(0, -1)
    minus = tuple(minus)

    return (
        data[single_slice(data.ndim, axis, 1, None)] +
        data[single_slice(data.ndim, axis, 0, -1)]
    ) / 2


def nodemap(shape, strides, periodic=(), init=0):
    indices = np.meshgrid(*(np.arange(s, dtype=int) for s in shape), indexing='ij')
    nodes = sum(i * s for i, s in zip(indices, strides)) + init
    for axis in periodic:
        nodes[single_index(nodes.ndim, axis, -1)] = nodes[single_index(nodes.ndim, axis, 0)]
    return nodes


def structured_cells(cellshape, pardim, nodemap=None):
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


def angle_mean_deg(data):
    data = np.deg2rad(data)
    return np.rad2deg(np.arctan2(np.mean(np.sin(data)), np.mean(np.cos(data))))


@contextmanager
def save_excursion(fp):
    assert fp.seekable()
    ptr = fp.tell()
    try:
        yield
    finally:
        fp.seek(ptr)


def split_commas(strings: Iterable[str]) -> Iterable[str]:
    for s in strings:
        yield from s.split(',')
