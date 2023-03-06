from __future__ import annotations

import logging
from io import BytesIO, StringIO
from typing import IO, Dict, Iterable, Iterator, List, Optional, Tuple, overload

import lrspline as lr
import numpy as np
import splipy.utils
from attrs import define
from numpy import floating, integer
from splipy import BSplineBasis, SplineObject
from splipy.io import G2
from typing_extensions import Self

from . import util
from .api import CellType, DiscreteTopology, Field, Tesselator, Topology
from .util import FieldData
from .zone import Coords


class UnstructuredTopology:
    celltype: CellType
    num_nodes: int
    cells: FieldData[integer]

    def __init__(self, num_nodes: int, cells: FieldData[integer], celltype: CellType):
        self.num_nodes = num_nodes
        self.cells = cells
        self.celltype = celltype

    @staticmethod
    def from_ifem(data: bytes) -> Tuple[Coords, UnstructuredTopology, FieldData[floating]]:
        io = BytesIO(data)

        first_line = next(io)
        assert first_line.startswith(b"# LAGRANGIAN")
        _, _, nodespec, elemspec, typespec = first_line.split()

        assert nodespec.startswith(b"nodes=")
        assert elemspec.startswith(b"elements=")
        assert typespec.startswith(b"type=")
        num_nodes = int(nodespec.split(b"=", 1)[1])
        num_cells = int(elemspec.split(b"=", 1)[1])
        celltype = typespec.split(b"=", 1)[1]
        assert celltype == b"hexahedron"

        nodes = np.zeros((num_nodes, 3), dtype=float)
        for i in range(num_nodes):
            nodes[i] = list(map(float, next(io).split()))

        cells = np.zeros((num_cells, 8), dtype=int)
        for i in range(num_cells):
            cells[i] = list(map(int, next(io).split()))
            cells[i, 6], cells[i, 7] = cells[i, 7], cells[i, 6]
            cells[i, 2], cells[i, 3] = cells[i, 3], cells[i, 2]

        corners = (tuple(nodes[0]),)
        topology = UnstructuredTopology(num_nodes, FieldData(cells), CellType.Hexahedron)
        return corners, topology, FieldData(nodes)

    @overload
    @staticmethod
    def join(other: Iterable[DiscreteTopology], /) -> UnstructuredTopology:
        ...

    @overload
    @staticmethod
    def join(*other: DiscreteTopology) -> UnstructuredTopology:
        ...

    @staticmethod
    def join(*other) -> UnstructuredTopology:
        iterable: Iterable[DiscreteTopology] = other if isinstance(other[0], DiscreteTopology) else other[0]
        num_nodes = 0
        celltype: Optional[CellType] = None

        def consume() -> Iterable[FieldData[integer]]:
            nonlocal num_nodes, celltype
            for topo in iterable:
                if celltype is None:
                    celltype = topo.celltype
                else:
                    assert celltype == topo.celltype
                yield topo.cells + num_nodes
                num_nodes += topo.num_nodes

        cells = FieldData.join(consume())
        assert celltype
        return UnstructuredTopology(
            num_nodes=num_nodes,
            cells=cells,
            celltype=celltype,
        )

    @property
    def pardim(self):
        return {CellType.Line: 1, CellType.Quadrilateral: 2, CellType.Hexahedron: 3}[self.celltype]

    @property
    def num_cells(self) -> int:
        return self.cells.ndofs

    def tesselator(self) -> NoopTesselator:
        return NoopTesselator()


class StructuredTopology:
    cellshape: Tuple[int, ...]
    celltype: CellType

    def __init__(self, cellshape: Tuple[int, ...], celltype: CellType):
        self.cellshape = cellshape
        self.celltype = celltype

    @property
    def pardim(self) -> int:
        return len(self.cellshape)

    @property
    def num_cells(self) -> int:
        return util.prod(self.cellshape)

    @property
    def num_nodes(self) -> int:
        return util.prod(s + 1 for s in self.cellshape)

    @property
    def cells(self) -> FieldData[integer]:
        return util.structured_cells(self.cellshape, self.pardim)

    def tesselator(self) -> Tesselator[Self]:
        return NoopTesselator()


class NoopTesselator(Tesselator[DiscreteTopology]):
    def tesselate_topology(self, topology: DiscreteTopology) -> DiscreteTopology:
        return topology

    def tesselate_field(
        self, topology: DiscreteTopology, field: Field, field_data: FieldData[floating]
    ) -> FieldData[floating]:
        return field_data


class G2Object(G2):
    def __init__(self, fstream: IO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == "w"
        super().__init__("")

    def __enter__(self) -> G2Object:
        return self


@define
class SplineTopology(Topology):
    bases: List[BSplineBasis]
    weights: Optional[np.ndarray]

    @staticmethod
    def from_splineobject(obj: SplineObject) -> Tuple[Coords, SplineTopology, FieldData[floating]]:
        corners = tuple(tuple(point) for point in obj.corners())
        if obj.rational:
            weights = util.transpose_butlast(obj.controlpoints[..., -1:]).flatten()
            cps = obj.controlpoints[..., :-1]
        else:
            weights = None
            cps = obj.controlpoints
        return (
            corners,
            SplineTopology(bases=obj.bases, weights=weights),
            FieldData(util.flatten_2d(util.transpose_butlast(cps))),
        )

    @staticmethod
    def from_bytes(data: bytes) -> Iterator[Tuple[Coords, SplineTopology, FieldData[floating]]]:
        yield from SplineTopology.from_string(data.decode())

    @staticmethod
    def from_string(data: str) -> Iterator[Tuple[Coords, SplineTopology, FieldData[floating]]]:
        with G2Object(StringIO(data), "r") as g2:
            for obj in g2.read():
                yield SplineTopology.from_splineobject(obj)

    @property
    def pardim(self) -> int:
        return len(self.bases)

    @property
    def num_nodes(self) -> int:
        return util.prod(basis.num_functions() for basis in self.bases)

    @property
    def num_cells(self) -> int:
        return util.prod(len(basis.knot_spans()) - 1 for basis in self.bases)

    def tesselator(self) -> Tesselator[Self]:
        return SplineTesselator(self, nvis=1)


class SplineTesselator(Tesselator[SplineTopology]):
    nodal_knots: List[np.ndarray]
    cellwise_knots: List[np.ndarray]

    def __init__(self, topology: SplineTopology, nvis: int = 1):
        self.nodal_knots = [util.subdivide_linear(basis.knot_spans(), nvis) for basis in topology.bases]

        self.cellwise_knots = [
            ((knots := np.array(basis.knot_spans()))[:-1] + knots[1:]) / 2 for basis in topology.bases
        ]

    def tesselate_topology(self, topology: SplineTopology) -> StructuredTopology:
        celltype = [CellType.Line, CellType.Quadrilateral, CellType.Hexahedron][len(self.nodal_knots) - 1]
        cellshape = tuple(len(knots) - 1 for knots in self.nodal_knots)
        return StructuredTopology(cellshape, celltype)

    def tesselate_field(
        self, topology: SplineTopology, field: Field, field_data: FieldData[floating]
    ) -> FieldData[floating]:
        if field.cellwise:
            bases = [BSplineBasis(order=1, knots=basis.knot_spans()) for basis in topology.bases]
            shape = tuple(basis.num_functions() for basis in bases)
            knots = self.cellwise_knots
            coeffs = field_data.data
            rational = False
        else:
            bases = topology.bases
            shape = tuple(basis.num_functions() for basis in topology.bases)
            knots = self.nodal_knots
            coeffs = field_data.data
            if topology.weights is not None:
                coeffs = np.hstack((coeffs, util.flatten_2d(topology.weights)))
            rational = topology.weights is not None

        coeffs = splipy.utils.reshape(coeffs, shape, order="F")
        new_spline = SplineObject(bases, coeffs, rational=rational, raw=True)
        return FieldData(util.flatten_2d(new_spline(*knots)))


@define
class LrTopology(Topology):
    obj: lr.LRSplineObject
    weights: Optional[np.ndarray]

    @staticmethod
    def from_lrobject(obj: lr.LRSplineObject) -> Tuple[Coords, LrTopology, FieldData[floating]]:
        corners = tuple(tuple(point) for point in obj.corners())
        rational = obj.dimension > obj.pardim
        if rational:
            logging.warning(
                f"Treating LR spline with parametric dimension {obj.pardim} "
                f"and physical dimension {obj.dimension} as rational"
            )

        if rational:
            weights = obj.controlpoints[:, -1]
            cps = obj.controlpoints[:, :-1]
        else:
            weights = None
            cps = obj.controlpoints

        return (
            corners,
            LrTopology(obj=obj, weights=weights),
            FieldData(cps),
        )

    @staticmethod
    def from_bytes(data: bytes) -> Iterator[Tuple[Coords, LrTopology, FieldData[floating]]]:
        yield from LrTopology.from_string(data.decode())

    @staticmethod
    def from_string(data: str) -> Iterator[Tuple[Coords, LrTopology, FieldData[floating]]]:
        for obj in lr.LRSplineObject.read_many(StringIO(data)):
            yield LrTopology.from_lrobject(obj)

    @property
    def pardim(self) -> int:
        return self.obj.pardim

    @property
    def num_nodes(self) -> int:
        return len(self.obj)

    @property
    def num_cells(self) -> int:
        return len(self.obj.elements)

    def tesselator(self) -> Tesselator[Self]:
        return LrTesselator(self.obj, self.weights, nvis=1)


class LrTesselator(Tesselator[LrTopology]):
    nodes: np.ndarray
    cells: FieldData[integer]
    weights: Optional[np.ndarray]
    nvis: int

    def __init__(self, obj: lr.LRSplineObject, weights: Optional[np.ndarray], nvis: int):
        nodes: Dict[Tuple[float, ...], int] = {}
        cells: List[List[int]] = []
        visitor = util.visit_face if obj.pardim == 2 else util.visit_volume
        for element in obj.elements:
            visitor(element, nodes, cells, nvis=1)
        self.nodes = FieldData.from_iter(nodes).numpy()
        self.cells = FieldData(np.array(cells, dtype=int))
        self.weights = weights
        self.nvis = nvis

    def tesselate_topology(self, topology: LrTopology) -> DiscreteTopology:
        celltype = CellType.Hexahedron if topology.pardim == 3 else CellType.Quadrilateral
        return UnstructuredTopology(len(self.nodes), self.cells, celltype)

    def tesselate_field(
        self, topology: LrTopology, field: Field, field_data: FieldData[floating]
    ) -> FieldData[floating]:
        if field.cellwise:
            cell_centers = (np.mean(self.nodes[c], axis=0) for c in self.cells.vectors)
            return FieldData.from_iter(
                field_data.numpy()[topology.obj.element_at(*c).id] for c in cell_centers
            )
        else:
            obj = topology.obj.clone()
            coeffs = field_data.data
            if self.weights is not None:
                coeffs = np.hstack((coeffs, self.weights))
            obj.controlpoints = coeffs
            evaluated = FieldData.from_iter(obj(*node) for node in self.nodes)
            if self.weights is not None:
                evaluated = evaluated.collapse_weights()
            return evaluated
