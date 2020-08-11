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
