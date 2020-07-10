from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO
from itertools import product

import lrspline as lr
import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis

from . import config
from .util import prod


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


class TensorTesselation:

    def __init__(self, patch):
        self.knots = tuple(subdivide_linear(knots, config.nvis) for knots in patch.knots())

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


class LRSurfaceTesselation:

    def __init__(self, patch):
        nodes, elements = OrderedDict(), []
        for el in patch.elements:
            subdivide_face(el, nodes, elements, config.nvis)

        self._nodes = nodes
        self._elements = np.array(elements, dtype=int)

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            ncomps = len(patch.elements) // coeffs.size
            coeffs = coeffs.reshape((-1, ncomps))
            coeffs = np.repeat(coeffs, config.nvis**2, axis=0)
            return coeffs

        if coeffs is not None:
            patch = patch.clone()
            patch.controlpoints = coeffs.reshape((len(patch.basis), -1))

        return np.array([patch(*node) for node in self._nodes], dtype=float)

    def elements(self):
        return 2, self._elements


class LRVolumeTesselation:

    def __init__(self, patch):
        nodes, elements = OrderedDict(), []
        for el in patch.elements:
            subdivide_volume(el, nodes, elements, config.nvis)

        self._nodes = nodes
        self._elements = np.array(elements, dtype=int)

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            ncomps = len(patch.elements) // coeffs.size
            coeffs = coeffs.reshape((-1, ncomps))
            coeffs = np.repeat(coeffs, config.nvis**3, axis=0)
            return coeffs

        if coeffs is not None:
            patch = patch.clone()
            patch.controlpoints = coeffs.reshape((len(patch.basis), -1))

        return np.array([patch(*node) for node in self._nodes], dtype=float)

    def elements(self):
        return 3, self._elements


class G2Object(splipy.io.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self


class Patch(ABC):

    @property
    @abstractmethod
    def num_physdim(self):
        """Number of physical dimensions."""
        pass

    @property
    @abstractmethod
    def num_pardim(self):
        """Number of parametric dimensions."""
        pass

    @property
    @abstractmethod
    def num_nodes(self):
        """Number of nodes."""
        pass

    @property
    @abstractmethod
    def num_cells(self):
        """Number of cells."""
        pass

    @property
    @abstractmethod
    def bounding_box(self):
        """Hashable bounding box."""
        pass

    @abstractmethod
    def tesselate(self):
        """Convert to a suitable discrete representation.
        Currently a UnstructuredPatch.
        """
        pass

    @abstractmethod
    def tesselate_field(self, coeffs, cells=False):
        """Convert a nodal or cell field to the same representation as
        returned by tesselate.
        """
        pass


class UnstructuredPatch(Patch):

    def __init__(self, nodes, cells):
        assert nodes.ndim == cells.ndim == 2
        self.nodes = nodes
        self.cells = cells

    @classmethod
    def from_lagrangian(cls, data):
        if isinstance(data, bytes):
            data = data.decode()
        assert isinstance(data, str)
        assert data.startswith('# LAGRANGIAN')
        lines = data.split('\n')
        specs, lines = lines[0][12:].split(), iter(lines[1:])

        # Decode nodes, elements, type
        assert specs[0].startswith('nodes=')
        nnodes = int(specs[0].split('=')[-1])
        assert specs[1].startswith('elements=')
        ncells = int(specs[1].split('=')[-1])
        assert specs[2].startswith('type=')
        celltype = specs[2].split('=')[-1]

        if celltype not in ('hexahedron',):
            raise ValueError("Unknown cell type: {}".format(celltype))

        # Read nodes and cells
        nodes = np.zeros((nnodes, 3))
        for i in range(nnodes):
            nodes[i] = list(map(float, next(lines).split()))

        cells = np.zeros((ncells, 8), dtype=np.int32)
        for i in range(ncells):
            cells[i] = list(map(int, next(lines).split()))
        cells[:,6], cells[:,7] = np.array(cells[:,7]), np.array(cells[:,6])
        cells[:,2], cells[:,3] = np.array(cells[:,3]), np.array(cells[:,2])

        return cls(nodes, cells)

    @property
    def num_physdim(self):
        return self.nodes.shape[-1]

    @property
    def num_pardim(self):
        return {
            (4, 8): 3
        }[self.num_physidm, self.cells.shape[-1]]

    @property
    def bounding_box(self):
        return tuple(
            (np.min(self.nodes[:,i]), np.max(self.nodes[:,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_cells(self):
        return len(self.cells)

    def tesselate(self):
        if config.nvis != 1:
            raise ValueError("Unstructured grid does not support tesselation with nvis > 1")
        return self

    def tesselate_field(self, coeffs, cells=False):
        if config.nvis != 1:
            raise ValueError("Unstructured grid does not support tesselation with nvis > 1")
        if cells:
            return coeffs.reshape((self.num_cells, -1))
        return coeffs.reshape((self.num_nodes, -1))


class LRPatch(Patch):

    def __init__(self, obj):
        if isinstance(obj, bytes):
            obj = obj.decode()
        if isinstance(obj, str):
            if obj.startswith('# LRSPLINE SURFACE'):
                obj = lr.LRSplineSurface(obj)
            elif obj.startswith('# LRSPLINE VOLUME'):
                obj = lr.LRSplineVolume(obj)
        assert isinstance(obj, lr.LRSplineObject)
        self.obj = obj

    @property
    def num_physdim(self):
        return self.obj.dimension

    @property
    def num_pardim(self):
        return self.obj.pardim

    @property
    def bounding_box(self):
        return tuple(
            (np.min(self.obj.controlpoints[:,i]), np.max(self.obj.controlpoints[:,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self):
        return len(self.obj)

    @property
    def num_cells(self):
        return len(self.obj.elements)

    @property
    def tesselator(self):
        if isinstance(self.obj, lr.LRSplineSurface):
            return LRSurfaceTesselation
        return LRVolumeTesselation

    def tesselate(self):
        tess = self.tesselator(self.obj)
        nodes = tess(self.obj)
        nodes = nodes.reshape((-1, nodes.shape[-1]))
        _, cells = tess.elements()
        return UnstructuredPatch(nodes, cells)

    def tesselate_field(self, coeffs, cells=False):
        tess = self.tesselator(self.obj)
        results = tess(self.obj, coeffs=coeffs, cells=cells)
        results = results.reshape((-1, results.shape[-1]))
        return results


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
    def num_physdim(self):
        return self.obj.dimension

    @property
    def num_pardim(self):
        return self.obj.pardim

    @property
    def bounding_box(self):
        return tuple(
            (np.min(self.obj.controlpoints[...,i]), np.max(self.obj.controlpoints[...,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self):
        return len(self.obj)

    @property
    def num_cells(self):
        return prod(len(k) - 1 for k in self.obj.knots())

    def tesselate(self):
        tess = TensorTesselation(self.obj)
        nodes = tess(self.obj)
        nodes = nodes.reshape((-1, nodes.shape[-1]))
        _, cells = tess.elements()
        return UnstructuredPatch(nodes, cells)

    def tesselate_field(self, coeffs, cells=False):
        tess = TensorTesselation(self.obj)
        results = tess(self.obj, coeffs=coeffs, cells=cells)
        results = results.reshape((-1, results.shape[-1]))
        return results
