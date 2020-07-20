from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO
from itertools import product

import lrspline as lr
import numpy as np
from singledispatchmethod import singledispatchmethod
import splipy.io
from splipy import SplineObject, BSplineBasis

from typing import Tuple, Any, Union, IO
from nptyping import NDArray, Float

from . import config
from .util import prod, flatten_2d


Array2D = NDArray[Any, Any]


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




# Abstract superclasses
# ----------------------------------------------------------------------


class Patch(ABC):

    @property
    @abstractmethod
    def num_physdim(self) -> int:
        """Number of physical dimensions."""
        pass

    @property
    @abstractmethod
    def num_pardim(self) -> int:
        """Number of parametric dimensions."""
        pass

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes."""
        pass

    @property
    @abstractmethod
    def num_cells(self) -> int:
        """Number of cells."""
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> Tuple[Tuple[float, float], ...]:
        """Hashable bounding box."""
        pass

    @abstractmethod
    def tesselate(self) -> 'Patch':
        """Convert to a suitable discrete representation.
        Currently an UnstructuredPatch.
        """
        pass

    @abstractmethod
    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        """Convert a nodal or cell field to the same representation as
        returned by tesselate.
        """
        pass


class Tesselator(ABC):

    def __init__(self, patch: Patch):
        self.source_patch = patch

    @abstractmethod
    def tesselate(self, patch: Patch) -> Patch:
        pass

    @abstractmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        pass


class UnstructuredPatch(Patch):
    """A patch that represents an unstructured collection of nodes and
    cells.  This is the lowest common grid form: all other grids
    should be convertable to it.
    """

    def __init__(self, nodes: Array2D, cells: Array2D):
        assert nodes.ndim == cells.ndim == 2
        self.nodes = nodes
        self.cells = cells

    @classmethod
    def from_lagrangian(cls, data: Union[bytes, str]) -> 'UnstructuredPatch':
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
    def num_physdim(self) -> int:
        return self.nodes.shape[-1]

    @property
    def num_pardim(self) -> int:
        return {
            (4, 8): 3
        }[self.num_physidm, self.cells.shape[-1]]

    @property
    def bounding_box(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(
            (np.min(self.nodes[:,i]), np.max(self.nodes[:,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def tesselate(self) -> Patch:
        if config.nvis != 1:
            raise ValueError("Unstructured grid does not support tesselation with nvis > 1")
        return self

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        if config.nvis != 1:
            raise ValueError("Unstructured grid does not support tesselation with nvis > 1")
        if cells:
            return coeffs.reshape((self.num_cells, -1))
        return coeffs.reshape((self.num_nodes, -1))



# LRSpline support
# ----------------------------------------------------------------------


class LRPatch(Patch):

    def __init__(self, obj: Union[bytes, str, lr.LRSplineObject]):
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
    def num_physdim(self) -> int:
        return self.obj.dimension

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

    @property
    def bounding_box(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(
            (np.min(self.obj.controlpoints[:,i]), np.max(self.obj.controlpoints[:,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self) -> int:
        return len(self.obj)

    @property
    def num_cells(self) -> int:
        return len(self.obj.elements)

    def tesselate(self) -> Patch:
        tess = LRTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = LRTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)


class LRTesselator(Tesselator):

    def __init__(self, patch: LRPatch):
        super().__init__(patch)
        nodes, cells = OrderedDict(), []
        subdivider = subdivide_face if patch.obj.pardim == 2 else subdivide_volume
        for el in patch.obj.elements:
            subdivider(el, nodes, cells, config.nvis)
        self.nodes = np.array(list(nodes))
        self.cells = np.array(cells, dtype=int)

    @singledispatchmethod
    def tesselate(self, patch: Patch) -> Patch:
        raise NotImplementedError

    @tesselate.register(LRPatch)
    def _(self, patch: LRPatch) -> Patch:
        spline = patch.obj
        nodes = np.array([spline(*node) for node in self.nodes], dtype=float)
        return UnstructuredPatch(nodes, self.cells)

    @singledispatchmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(LRPatch)
    def _(self, patch: LRPatch, coeffs: Array2D, cells: bool = False) -> Array2D:
        spline = patch.obj

        if not cells:
            # Create a new patch with substituted control points, and
            # evaluate it at the predetermined knot values.
            newspline = spline.clone()
            newspline.controlpoints = coeffs.reshape((len(spline), -1))
            return np.array([newspline(*node) for node in self.nodes], dtype=float)

        # For every cell center, check which cell it belongs to in the
        # reference spline, then use that coefficient.
        coeffs = flatten_2d(coeffs)
        cell_centers = [np.mean(self.nodes[c,:], axis=0) for c in self.cells]
        return np.array([coeffs[spline.element_at(*c).id, :] for c in cell_centers])



# Splipy support
# ----------------------------------------------------------------------


class G2Object(splipy.io.G2):
    """G2 reader subclass to allow reading from a stream."""

    def __init__(self, fstream: IO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self) -> 'G2Object':
        return self


class SplinePatch(Patch):
    """A representation of a Splipy SplineObject."""

    def __init__(self, obj: Union[bytes, str, SplineObject]):
        if isinstance(obj, bytes):
            obj = obj.decode()
        if isinstance(obj, str):
            g2data = StringIO(obj)
            with G2Object(g2data, 'r') as g:
                obj = g.read()[0]
        assert isinstance(obj, SplineObject)
        self.obj = obj

    @property
    def num_physdim(self) -> int:
        return self.obj.dimension

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

    @property
    def bounding_box(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(
            (np.min(self.obj.controlpoints[...,i]), np.max(self.obj.controlpoints[...,i]))
            for i in range(self.num_physdim)
        )

    @property
    def num_nodes(self) -> int:
        return len(self.obj)

    @property
    def num_cells(self) -> int:
        return prod(len(k) - 1 for k in self.obj.knots())

    def tesselate(self) -> Patch:
        tess = TensorTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = TensorTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)


class TensorTesselator(Tesselator):

    def __init__(self, patch: SplinePatch):
        super().__init__(patch)
        knots = patch.obj.knots()
        self.knots = tuple(subdivide_linear(kts, config.nvis) for kts in knots)

    @singledispatchmethod
    def tesselate(self, patch: Patch) -> Patch:
        raise NotImplementedError

    @tesselate.register(SplinePatch)
    def _(self, patch: SplinePatch):
        nodes = flatten_2d(patch.obj(*self.knots))
        return UnstructuredPatch(nodes, self.cells())

    @singledispatchmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(SplinePatch)
    def _(self, patch: SplinePatch, coeffs: Array2D, cells: bool = False) -> Array2D:
        spline = patch.obj

        if not cells:
            # Create a new patch with substituted control points, and
            # evaluate it at the predetermined knot values.
            coeffs = splipy.utils.reshape(coeffs, spline.shape, order='F')
            if spline.rational:
                coeffs = np.concatenate((coeffs, spline.controlpoints[..., -1, np.newaxis]), axis=-1)
            newspline = SplineObject(spline.bases, coeffs, spline.rational, raw=True)
            knots = self.knots

        else:
            # Create a piecewise constant spline object, and evaluate
            # it in cell centers.
            bases = [BSplineBasis(1, kts) for kts in spline.knots()]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            newspline = SplineObject(bases, coeffs, False, raw=True)
            knots = [[(a+b)/2 for a, b in zip(t[:-1], t[1:])] for t in self.knots]

        return flatten_2d(newspline(*knots))

    def cells(self):
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

        return eidxs
