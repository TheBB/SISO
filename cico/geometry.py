from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO

import lrspline as lr
import numpy as np
from singledispatchmethod import singledispatchmethod
import splipy.io
from splipy import SplineObject, BSplineBasis
import treelog as log

from typing import Tuple, Any, Union, IO, Dict, Hashable, List
from .typing import Array2D, BoundingBox, PatchID, Shape

from . import config

from .util import (
    prod, flatten_2d, ensure_ncomps,
    subdivide_face, subdivide_linear, subdivide_volume,
    structured_cells,
)



# Abstract superclasses
# ----------------------------------------------------------------------


class CellType:

    num_nodes: int
    num_pardim: int
    structured: bool


class Quad(CellType):

    num_nodes = 4
    num_pardim = 2
    structured = True


class Hex(CellType):

    num_nodes = 8
    num_pardim = 3
    structured = True



# Abstract superclasses
# ----------------------------------------------------------------------


class Patch(ABC):

    key: PatchID

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
    def bounding_box(self) -> BoundingBox:
        """Hashable bounding box."""
        pass

    @abstractmethod
    def tesselate(self) -> 'UnstructuredPatch':
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

    @abstractmethod
    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
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



# Unstructured and structured support
# ----------------------------------------------------------------------


class UnstructuredPatch(Patch):
    """A patch that represents an unstructured collection of nodes and
    cells.  This is the lowest common grid form: all other grids
    should be convertable to it.
    """

    nodes: Array2D
    celltype: CellType

    _cells: Array2D

    def __init__(self, key: PatchID, nodes: Array2D, cells: Array2D, celltype: CellType):
        assert nodes.ndim == cells.ndim == 2
        self.key = key
        self.nodes = nodes
        self._cells = cells
        self.celltype = celltype
        assert cells.shape[-1] == celltype.num_nodes

    @property
    def cells(self) -> Array2D:
        return self._cells

    @classmethod
    def from_lagrangian(cls, key: PatchID, data: Union[bytes, str]) -> 'UnstructuredPatch':
        if isinstance(data, bytes):
            data = data.decode()
        assert isinstance(data, str)
        assert data.startswith('# LAGRANGIAN')
        all_lines = data.split('\n')
        specs, lines = all_lines[0][12:].split(), iter(all_lines[1:])

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

        return cls(key, nodes, cells, celltype=Hex())

    @property
    def num_physdim(self) -> int:
        return self.nodes.shape[-1]

    @property
    def num_pardim(self) -> int:
        return self.celltype.num_pardim

    @property
    def bounding_box(self) -> BoundingBox:
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

    def tesselate(self) -> 'UnstructuredPatch':
        return self

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        if cells:
            return coeffs.reshape((self.num_cells, -1))
        return coeffs.reshape((self.num_nodes, -1))

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        self.nodes = ensure_ncomps(self.nodes, ncomps, allow_scalar)


class StructuredPatch(UnstructuredPatch):
    """A patch that represents an structured collection of nodes and
    cells.  This is interchangeable with UnstructuredPatch
    """

    shape: Shape

    def __init__(self, key: PatchID, nodes: Array2D, shape: Shape, celltype: CellType):
        self.key = key
        self.nodes = nodes
        self.celltype = celltype
        self.shape = shape
        assert celltype.structured
        assert len(shape) == celltype.num_pardim
        assert prod(k+1 for k in shape) == len(nodes)

    @property
    def cells(self) -> Array2D:
        return structured_cells(self.shape, self.num_pardim)

    @property
    def num_cells(self) -> int:
        return prod(self.shape)



# LRSpline support
# ----------------------------------------------------------------------


class LRPatch(Patch):

    def __init__(self, key: PatchID, obj: Union[bytes, str, lr.LRSplineObject]):
        if isinstance(obj, bytes):
            obj = obj.decode()
        if isinstance(obj, str):
            if obj.startswith('# LRSPLINE SURFACE'):
                obj = lr.LRSplineSurface(obj)
            elif obj.startswith('# LRSPLINE VOLUME'):
                obj = lr.LRSplineVolume(obj)
        assert isinstance(obj, lr.LRSplineObject)
        self.obj = obj
        self.key = key

    @property
    def num_physdim(self) -> int:
        return self.obj.dimension

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

    @property
    def bounding_box(self) -> BoundingBox:
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

    def tesselate(self) -> UnstructuredPatch:
        tess = LRTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = LRTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        self.obj.controlpoints = ensure_ncomps(self.obj.controlpoints, ncomps, allow_scalar)


class LRTesselator(Tesselator):

    def __init__(self, patch: LRPatch):
        super().__init__(patch)
        nodes: Dict[Tuple[float, ...], int] = dict()
        cells: List[List[int]] = []
        subdivider = subdivide_face if patch.obj.pardim == 2 else subdivide_volume
        for el in patch.obj.elements:
            subdivider(el, nodes, cells, config.nvis)
        self.nodes = np.array(list(nodes))
        self.cells = np.array(cells, dtype=int)

    @singledispatchmethod
    def tesselate(self, patch: Patch) -> UnstructuredPatch:
        raise NotImplementedError

    @tesselate.register(LRPatch)
    def _1(self, patch: LRPatch) -> UnstructuredPatch:
        spline = patch.obj
        nodes = np.array([spline(*node) for node in self.nodes], dtype=float)
        celltype = Hex() if patch.num_pardim == 3 else Quad()
        return UnstructuredPatch((*patch.key, 'tesselated'), nodes, self.cells, celltype=celltype)

    @singledispatchmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(LRPatch)
    def _2(self, patch: LRPatch, coeffs: Array2D, cells: bool = False) -> Array2D:
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

    def __init__(self, key: PatchID, obj: Union[bytes, str, SplineObject]):
        if isinstance(obj, bytes):
            obj = obj.decode()
        if isinstance(obj, str):
            g2data = StringIO(obj)
            with G2Object(g2data, 'r') as g:
                obj = g.read()[0]
        assert isinstance(obj, SplineObject)
        self.obj = obj
        self.key = key

    @property
    def num_physdim(self) -> int:
        return self.obj.dimension

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

    @property
    def bounding_box(self) -> BoundingBox:
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

    def tesselate(self) -> UnstructuredPatch:
        tess = TensorTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = TensorTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        if allow_scalar and self.obj.dimension == 1:
            return
        self.obj.set_dimension(ncomps)


class TensorTesselator(Tesselator):

    def __init__(self, patch: SplinePatch):
        super().__init__(patch)
        knots = patch.obj.knots()
        self.knots = list(subdivide_linear(kts, config.nvis) for kts in knots)

    @singledispatchmethod
    def tesselate(self, patch: Patch) -> Patch:
        raise NotImplementedError

    @tesselate.register(SplinePatch)
    def _1(self, patch: SplinePatch) -> UnstructuredPatch:
        nodes = flatten_2d(patch.obj(*self.knots))
        celltype = Hex() if patch.num_pardim == 3 else Quad()
        cellshape = tuple(len(kts) - 1 for kts in self.knots)
        return StructuredPatch((*patch.key, 'tesselated'), nodes, cellshape, celltype=celltype)

    @singledispatchmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(SplinePatch)
    def _2(self, patch: SplinePatch, coeffs: Array2D, cells: bool = False) -> Array2D:
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



# GeometryManager
# ----------------------------------------------------------------------


class GeometryManager:

    patch_keys: Dict[PatchID, int]
    bounding_boxes: Dict[BoundingBox, int]

    def __init__(self):
        self.patch_keys = dict()
        self.bounding_boxes = dict()

    def update(self, patch: Patch):
        if patch.key not in self.patch_keys:
            patchid = len(self.patch_keys)
            log.debug(f"New unique patch detected, assigned ID {patchid}")
            self.patch_keys[patch.key] = patchid
        else:
            patchid = self.patch_keys[patch.key]
        self.bounding_boxes[patch.bounding_box] = patchid
        return patchid

    def global_id(self, patch: Patch):
        try:
            return self.bounding_boxes[patch.bounding_box]
        except KeyError:
            log.error("Unable to find corresponding geometry patch")
            return None
