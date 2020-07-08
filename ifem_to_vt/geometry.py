from abc import ABC, abstractmethod
from io import StringIO
from itertools import product

import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis

from .util import prod


def subdivide_linear(knots, nvis):
    out = []
    for left, right in zip(knots[:-1], knots[1:]):
        out.extend(np.linspace(left, right, num=nvis, endpoint=False))
    out.append(knots[-1])
    return out


class TensorTesselation:

    def __init__(self, patch, nvis=1):
        self.knots = tuple(subdivide_linear(knots, nvis) for knots in patch.knots())

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            bases = [BSplineBasis(1, kts) for kts in patch.knots()]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            patch = SplineObject(bases, coeffs, False, raw=True)
            knots = [[(a+b)/2 for a, b in zip(t[:-1], t[1:])] for t in self.knots]
            return patch(*knots)

        if coeffs is not None:
            coeffs = splipy.utils.reshape(coeffs, patch.shape, order='F')
            if patch.rational:
                coeffs = np.concatenate((coeffs, patch.controlpoints[..., -1, np.newaxis]), axis=-1)
            patch = SplineObject(patch.bases, coeffs, patch.rational, raw=True)
        return patch(*self.knots)

    def elements(self):
        nshape = tuple(len(k) for k in self.knots)
        ranges = [range(k-1) for k in nshape]
        nidxs = [np.array(q) for q in zip(*product(*ranges))]
        eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)), dtype=int)
        if len(nidxs) == 1:
            eidxs[:,0] = nidxs[0]
            eidxs[:,1] = nidxs[0] + 1
        elif len(nidxs) == 2:
            i, j = nidxs
            eidxs[:,0] = np.ravel_multi_index((i, j), nshape)
            eidxs[:,1] = np.ravel_multi_index((i+1, j), nshape)
            eidxs[:,2] = np.ravel_multi_index((i+1, j+1), nshape)
            eidxs[:,3] = np.ravel_multi_index((i, j+1), nshape)
        elif len(nidxs) == 3:
            i, j, k = nidxs
            eidxs[:,0] = np.ravel_multi_index((i, j, k), nshape)
            eidxs[:,1] = np.ravel_multi_index((i+1, j, k), nshape)
            eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nshape)
            eidxs[:,3] = np.ravel_multi_index((i, j+1, k), nshape)
            eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nshape)
            eidxs[:,5] = np.ravel_multi_index((i+1, j, k+1), nshape)
            eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nshape)
            eidxs[:,7] = np.ravel_multi_index((i, j+1, k+1), nshape)

        return len(nidxs), eidxs


class G2Object(splipy.io.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self


class Patch(ABC):
    pass


class UnstructuredPatch(Patch):

    def __init__(self, nodes, cells):
        assert nodes.ndim == cells.ndim == 2
        self.nodes = nodes
        self.cells = cells

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_cells(self):
        return len(self.cells)


class SplinePatch(Patch):

    def __init__(self, obj):
        if isinstance(obj, bytes):
            obj = obj.decode()
        if isinstance(obj, str):
            g2data = StringIO(obj)
            with G2Object(g2data, 'r') as g:
                obj = g.read()[0]
        assert isinstance(obj, SplineObject)
        self.obj = obj

    @property
    def key(self):
        return tuple(tuple(p) for p in self.obj.corners())

    @property
    def num_nodes(self):
        return len(self.obj)

    @property
    def num_cells(self):
        return prod(len(k) - 1 for k in self.obj.knots())

    def tesselate(self, nvis=1):
        tess = TensorTesselation(self.obj, nvis=nvis)
        nodes = tess(self.obj)
        nodes = nodes.reshape((-1, nodes.shape[-1]))
        _, cells = tess.elements()
        return UnstructuredPatch(nodes, cells)

    def tesselate_coeffs(self, coeffs, cells=False, nvis=1):
        tess = TensorTesselation(self.obj, nvis=nvis)
        results = tess(self.obj, coeffs=coeffs, cells=cells)
        if results.ndim > 1:
            results = results.reshape((-1, results.shape[-1]))
        return results
