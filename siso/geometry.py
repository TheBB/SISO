from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO

import lrspline as lr
import numpy as np
from singledispatchmethod import singledispatchmethod
import splipy.io
from splipy import SplineObject, BSplineBasis
import treelog as log

from typing import Tuple, Any, Union, IO, Dict, List, Iterable, Optional
from .typing import Array1D, Array2D, PatchKey, Shape, Knots

from . import config

from .util import (
    prod, flatten_2d, ensure_ncomps,
    subdivide_face, subdivide_linear, subdivide_volume,
    structured_cells, transpose_butlast
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

    key: PatchKey

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
    _num_nodes: int

    cells: Array2D

    def __init__(self, key: PatchKey, num_nodes: int, cells: Array2D, celltype: CellType):
        self.key = key
        self.cells = cells
        self._num_nodes = num_nodes
        self.celltype = celltype
        assert cells.shape[-1] == celltype.num_nodes

    @classmethod
    def from_lagrangian(cls, key: PatchKey, data: Union[bytes, str]) -> Tuple['UnstructuredPatch', Array2D]:
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

        return cls(key, nnodes, cells, celltype=Hex()), nodes

    @property
    def num_pardim(self) -> int:
        return self.celltype.num_pardim

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def tesselate(self) -> 'UnstructuredPatch':
        return self

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        return coeffs


class StructuredPatch(UnstructuredPatch):
    """A patch that represents an structured collection of nodes and
    cells.  This is interchangeable with UnstructuredPatch
    """

    shape: Shape

    def __init__(self, key: PatchKey, shape: Shape, celltype: CellType):
        self.key = key
        self.celltype = celltype
        self.shape = shape
        self._num_nodes = prod(k+1 for k in shape)
        assert celltype.structured
        assert len(shape) == celltype.num_pardim

    @property
    def cells(self) -> Array2D:
        return structured_cells(self.shape, self.num_pardim)

    @property
    def num_cells(self) -> int:
        return prod(self.shape)



# LRSpline support
# ----------------------------------------------------------------------


class LRPatch(Patch):

    obj: lr.LRSplineObject

    def __init__(self, key: PatchKey, obj: lr.LRSplineObject):
        self.obj = obj
        self.key = key

    @classmethod
    def from_string(cls, key: PatchKey, data: Union[bytes, str]) -> Iterable[Tuple['LRPatch', Array2D]]:
        if isinstance(data, bytes):
            data = data.decode()
        data = StringIO(data)
        for i, obj in enumerate(lr.LRSplineObject.read_many(data)):
            cps = obj.controlpoints.reshape(-1, obj.dimension)
            obj.dimension = 0
            yield cls((*key, i), obj), cps

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

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
    def _(self, patch: LRPatch) -> UnstructuredPatch:
        spline = patch.obj
        celltype = Hex() if patch.num_pardim == 3 else Quad()
        return UnstructuredPatch((*patch.key, 'tesselated'), len(self.nodes), self.cells, celltype=celltype)

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

    bases: List[BSplineBasis]
    weights: Optional[Array1D]

    def __init__(self, key: PatchKey, bases: List[BSplineBasis], weights: Optional[Array1D] = None):
        self.key = key
        self.bases = bases
        self.weights = weights

    @classmethod
    def from_string(cls, key: PatchKey, data: Union[bytes, str]) -> Iterable[Tuple['SplinePatch', Array2D]]:
        if isinstance(data, bytes):
            data = data.decode()
        g2data = StringIO(data)
        with G2Object(g2data, 'r') as g:
            for i, obj in enumerate(g.read()):
                cps = flatten_2d(transpose_butlast(obj.controlpoints))
                weights = None
                if obj.rational:
                    weights = cps[:, -1]
                    cps = cps[:, :-1]
                yield cls((*key, i), obj.bases, weights), cps

    @property
    def num_pardim(self) -> int:
        return len(self.bases)

    @property
    def num_nodes(self) -> int:
        return prod(self.nodeshape)

    @property
    def num_cells(self) -> int:
        return prod(len(kts) - 1 for kts in self.knots)

    @property
    def knots(self) -> Knots:
        return tuple(b.knot_spans() for b in self.bases)

    @property
    def nodeshape(self) -> Shape:
        return tuple(b.num_functions() for b in self.bases)

    @property
    def rational(self) -> bool:
        return self.weights is not None

    def tesselate(self) -> UnstructuredPatch:
        tess = TensorTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = TensorTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)


class TensorTesselator(Tesselator):

    def __init__(self, patch: SplinePatch):
        super().__init__(patch)
        self.knots = list(subdivide_linear(b.knot_spans(), config.nvis) for b in patch.bases)

    @singledispatchmethod
    def tesselate(self, patch: Patch) -> Patch:
        raise NotImplementedError

    @tesselate.register(SplinePatch)
    def _1(self, patch: SplinePatch) -> UnstructuredPatch:
        celltype = Hex() if patch.num_pardim == 3 else Quad()
        cellshape = tuple(len(kts) - 1 for kts in self.knots)
        return StructuredPatch((*patch.key, 'tesselated'), cellshape, celltype=celltype)

    @singledispatchmethod
    def tesselate_field(self, patch: Patch, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(SplinePatch)
    def _(self, patch: SplinePatch, coeffs: Array2D, cells: bool = False) -> Array2D:
        if not cells:
            # Create a new patch with substituted control points, and
            # evaluate it at the predetermined knot values.
            if patch.weights is not None:
                coeffs = np.concatenate((coeffs, flatten_2d(patch.weights)), axis=-1)
            coeffs = splipy.utils.reshape(coeffs, patch.nodeshape, order='F')
            newspline = SplineObject(patch.bases, coeffs, patch.rational, raw=True)
            knots = self.knots

        else:
            # Create a piecewise constant spline object, and evaluate
            # it in cell centers.
            bases = [BSplineBasis(1, b.knot_spans()) for b in patch.bases]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            newspline = SplineObject(bases, coeffs, False, raw=True)
            knots = [[(a+b)/2 for a, b in zip(t[:-1], t[1:])] for t in self.knots]

        return flatten_2d(newspline(*knots))



# GeometryManager
# ----------------------------------------------------------------------


class GeometryManager:

    patch_keys: Dict[PatchKey, int]

    def __init__(self):
        self.patch_keys = dict()

    def id_by_key(self, key: PatchKey):
        while key:
            try:
                return self.patch_keys[key]
            except KeyError:
                key = key[:-1]
        raise KeyError

    def update(self, patch: Patch, data: Array2D) -> int:
        try:
            patchid = self.id_by_key(patch.key)
        except KeyError:
            patchid = len(self.patch_keys)
            log.debug(f"New unique patch detected, assigned ID {patchid}")
            self.patch_keys[patch.key] = patchid
        return patchid

    def global_id(self, patch: Patch) -> int:
        try:
            return self.id_by_key(patch.key)
        except KeyError:
            raise ValueError(f"Unable to find corresponding geometry patch for {patch.key}")
