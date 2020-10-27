from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO

from dataclasses import dataclass
import lrspline as lr
import numpy as np
from singledispatchmethod import singledispatchmethod
import splipy.io
from splipy import SplineObject, BSplineBasis
import treelog as log

from typing import Tuple, Any, Union, IO, Dict, List, Iterable, Optional, BinaryIO
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

    def __eq__(self, other):
        if not isinstance(other, CellType):
            return NotImplemented
        return type(self) == type(other)


class Line(CellType):

    num_nodes = 2
    num_pardim = 1
    structured = True


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


@dataclass
class Patch:

    key: PatchKey
    topology: Optional['Patch'] = None


class Topology(ABC):

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
    def tesselate(self) -> 'UnstructuredTopology':
        """Convert to a suitable discrete representation.
        Currently an UnstructuredTopology.
        """
        pass

    @abstractmethod
    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        """Convert a nodal or cell field to the same representation as
        returned by tesselate.
        """
        pass


class Tesselator(ABC):

    def __init__(self, topo: Topology):
        self.source_topo = topo

    @abstractmethod
    def tesselate(self, topo: Topology) -> Topology:
        pass

    @abstractmethod
    def tesselate_field(self, topo: Topology, coeffs: Array2D, cells: bool = False) -> Array2D:
        pass



# Unstructured and structured support
# ----------------------------------------------------------------------


class UnstructuredTopology(Topology):
    """A topology that represents an unstructured collection of nodes
    and cells.  This is the lowest common grid form: all other grids
    should be convertable to it.
    """

    nodes: Array2D
    celltype: CellType
    _num_nodes: int

    cells: Array2D

    def __init__(self, num_nodes: int, cells: Array2D, celltype: CellType):
        self.cells = cells
        self._num_nodes = num_nodes
        self.celltype = celltype
        assert cells.shape[-1] == celltype.num_nodes

    @classmethod
    def join(cls, left: 'UnstructuredTopology', right: 'UnstructuredTopology') -> 'UnstructuredTopology':
        assert left.celltype == right.celltype
        return cls(
            left.num_nodes + right.num_nodes,
            np.vstack((left.cells, right.cells + left.num_nodes)),
            left.celltype
        )

    @classmethod
    def from_lagrangian(cls, data: BinaryIO) -> Tuple['UnstructuredTopology', Array2D]:
        first_line = next(data)
        assert first_line.startswith(b'# LAGRANGIAN')
        specs = first_line.split()[2:]

        # Decode nodes, elements, type
        assert specs[0].startswith(b'nodes=')
        nnodes = int(specs[0].split(b'=')[-1])
        assert specs[1].startswith(b'elements=')
        ncells = int(specs[1].split(b'=')[-1])
        assert specs[2].startswith(b'type=')
        celltype = specs[2].split(b'=')[-1]

        if celltype not in (b'hexahedron',):
            raise ValueError("Unknown cell type: {}".format(celltype.decode()))

        # Read nodes and cells
        nodes = np.zeros((nnodes, 3))
        for i in range(nnodes):
            nodes[i] = list(map(float, next(data).split()))

        cells = np.zeros((ncells, 8), dtype=np.int32)
        for i in range(ncells):
            cells[i] = list(map(int, next(data).split()))
            cells[i,6], cells[i,7] = cells[i,7], cells[i,6]
            cells[i,2], cells[i,3] = cells[i,3], cells[i,2]

        return cls(nnodes, cells, celltype=Hex()), nodes

    @property
    def num_pardim(self) -> int:
        return self.celltype.num_pardim

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def tesselate(self) -> 'UnstructuredTopology':
        return self

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        return coeffs


class StructuredTopology(UnstructuredTopology):
    """A topology that represents an structured collection of nodes
    and cells.  This is interchangeable with UnstructuredTopology.
    """

    shape: Shape

    def __init__(self, shape: Shape, celltype: CellType):
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


class LRTopology(Topology):

    obj: lr.LRSplineObject

    def __init__(self, obj: lr.LRSplineObject):
        self.obj = obj

    @classmethod
    def from_string(cls, data: Union[bytes, str]) -> Iterable[Tuple['LRTopology', Array2D]]:
        if isinstance(data, bytes):
            data = data.decode()
        data = StringIO(data)
        for i, obj in enumerate(lr.LRSplineObject.read_many(data)):
            cps = obj.controlpoints.reshape(-1, obj.dimension)
            obj.dimension = 0
            yield cls(obj), cps

    @property
    def num_pardim(self) -> int:
        return self.obj.pardim

    @property
    def num_nodes(self) -> int:
        return len(self.obj)

    @property
    def num_cells(self) -> int:
        return len(self.obj.elements)

    def tesselate(self) -> UnstructuredTopology:
        tess = LRTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = LRTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)


class LRTesselator(Tesselator):

    def __init__(self, topo: LRTopology):
        super().__init__(topo)
        nodes: Dict[Tuple[float, ...], int] = dict()
        cells: List[List[int]] = []
        subdivider = subdivide_face if topo.obj.pardim == 2 else subdivide_volume
        for el in topo.obj.elements:
            subdivider(el, nodes, cells, config.nvis)
        self.nodes = np.array(list(nodes))
        self.cells = np.array(cells, dtype=int)

    @singledispatchmethod
    def tesselate(self, topo: Topology) -> UnstructuredTopology:
        raise NotImplementedError

    @tesselate.register(LRTopology)
    def _(self, patch: LRTopology) -> UnstructuredTopology:
        spline = patch.obj
        celltype = Hex() if patch.num_pardim == 3 else Quad()
        return UnstructuredTopology(len(self.nodes), self.cells, celltype=celltype)

    @singledispatchmethod
    def tesselate_field(self, topo: Topology, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(LRTopology)
    def _(self, patch: LRTopology, coeffs: Array2D, cells: bool = False) -> Array2D:
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


class SplineTopology(Topology):
    """A representation of a Splipy SplineObject."""

    bases: List[BSplineBasis]
    weights: Optional[Array1D]

    def __init__(self, bases: List[BSplineBasis], weights: Optional[Array1D] = None):
        self.bases = bases
        self.weights = weights

    @classmethod
    def from_string(cls, data: Union[bytes, str]) -> Iterable[Tuple['SplinePatch', Array2D]]:
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
                yield cls(obj.bases, weights), cps

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

    def tesselate(self) -> UnstructuredTopology:
        tess = TensorTesselator(self)
        return tess.tesselate(self)

    def tesselate_field(self, coeffs: Array2D, cells: bool = False) -> Array2D:
        tess = TensorTesselator(self)
        return tess.tesselate_field(self, coeffs, cells=cells)


class TensorTesselator(Tesselator):

    def __init__(self, topo: SplineTopology):
        super().__init__(topo)
        self.knots = list(subdivide_linear(b.knot_spans(), config.nvis) for b in topo.bases)

    @singledispatchmethod
    def tesselate(self, topo: Topology) -> Topology:
        raise NotImplementedError

    @tesselate.register(SplineTopology)
    def _1(self, topo: SplineTopology) -> UnstructuredTopology:
        celltype = {
            1: Line(),
            2: Quad(),
            3: Hex()
        }[topo.num_pardim]
        cellshape = tuple(len(kts) - 1 for kts in self.knots)
        return StructuredTopology(cellshape, celltype=celltype)

    @singledispatchmethod
    def tesselate_field(self, topo: Topology, coeffs: Array2D, cells: bool = False) -> Array2D:
        raise NotImplementedError

    @tesselate_field.register(SplineTopology)
    def _(self, topo: SplineTopology, coeffs: Array2D, cells: bool = False) -> Array2D:
        if not cells:
            # Create a new patch with substituted control points, and
            # evaluate it at the predetermined knot values.
            if topo.weights is not None:
                coeffs = np.concatenate((coeffs, flatten_2d(topo.weights)), axis=-1)
            coeffs = splipy.utils.reshape(coeffs, topo.nodeshape, order='F')
            newspline = SplineObject(topo.bases, coeffs, topo.rational, raw=True)
            knots = self.knots

        else:
            # Create a piecewise constant spline object, and evaluate
            # it in cell centers.
            bases = [BSplineBasis(1, b.knot_spans()) for b in topo.bases]
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

    def update(self, key: PatchKey, data: Array2D) -> int:
        try:
            patchid = self.id_by_key(key)
        except KeyError:
            patchid = len(self.patch_keys)
            log.debug(f"New unique patch detected {key}, assigned ID {patchid}")
            self.patch_keys[key] = patchid
        return patchid

    def global_id(self, key: PatchKey) -> int:
        try:
            return self.id_by_key(key)
        except KeyError:
            msg = f"Unable to find corresponding geometry patch for {key}"
            if config.strict_id:
                msg += f", consider trying without {config.cname('strict_id')}"
            raise ValueError(msg)
