from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from splipy import BSplineBasis, SplineObject
from splipy.io import G2
import splipy.utils

import numpy as np

from .api import Field, Topology, DiscreteTopology, Tesselator, CellType
from .field import FieldData
from .zone import Coords
from . import util

from typing_extensions import Self
from typing import (
    IO,
    Iterator,
    List,
    Optional,
    Tuple,
)


class UnstructuredTopology:
    celltype: CellType
    num_nodes: int
    cells: np.ndarray

    def __init__(self, num_nodes: int, cells: np.ndarray, celltype: CellType):
        self.num_nodes = num_nodes
        self.cells = cells
        self.celltype = celltype

    @staticmethod
    def join(left: DiscreteTopology, right: DiscreteTopology) -> UnstructuredTopology:
        assert left.celltype == right.celltype
        return UnstructuredTopology(
            num_nodes=left.num_nodes + right.num_nodes,
            cells=np.vstack((left.cells, right.cells + left.num_nodes)),
            celltype=left.celltype,
        )

    @property
    def pardim(self):
        return {
            CellType.Line: 1,
            CellType.Quadrilateral: 2,
            CellType.Hexahedron: 3
        }[self.celltype]

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def tesselator(self) -> NoopTesselator:
        return NoopTesselator()


class StructuredTopology:
    cellshape: Tuple[int, ...]
    celltype: CellType

    def __init__(self, shape: Tuple[int, ...], celltype: CellType):
        self.cellshape = shape
        self.celltype = celltype

    @property
    def pardim(self) -> int:
        return len(self.cellshape)

    def tesselator(self) -> Tesselator[Self]:
        raise NotImplementedError

    @property
    def num_cells(self) -> int:
        return util.prod(self.cellshape)

    @property
    def num_nodes(self) -> int:
        return util.prod(s + 1 for s in self.cellshape)

    @property
    def cells(self) -> np.ndarray:
        return util.structured_cells(self.cellshape, self.pardim)


class NoopTesselator(Tesselator[DiscreteTopology]):
    def tesselate_topology(self, topology: DiscreteTopology) -> DiscreteTopology:
        return topology

    def tesselate_field(self, topology: DiscreteTopology, field: Field, field_data: FieldData) -> FieldData:
        return field_data


class G2Object(G2):
    def __init__(self, fstream: IO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super().__init__('')

    def __enter__(self) -> G2Object:
        return self


@dataclass
class SplineTopology(Topology):
    bases: List[BSplineBasis]
    weights: Optional[np.ndarray]

    @staticmethod
    def from_splineobject(obj: SplineObject) -> Tuple[Coords, SplineTopology, FieldData]:
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
            FieldData(data=util.flatten_2d(util.transpose_butlast(cps))),
        )

    @staticmethod
    def from_bytes(data: bytes) -> Iterator[Tuple[Coords, SplineTopology, FieldData]]:
        yield from SplineTopology.from_string(data.decode())

    @staticmethod
    def from_string(data: str) -> Iterator[Tuple[Coords, SplineTopology, FieldData]]:
        with G2Object(StringIO(data), 'r') as g2:
            for obj in g2.read():
                yield SplineTopology.from_splineobject(obj)

    @property
    def pardim(self) -> int:
        return len(self.bases)

    def tesselator(self) -> Tesselator[Self]:
        return SplineTesselator(self, nvis=1)


class SplineTesselator(Tesselator[SplineTopology]):
    nodal_knots: List[np.ndarray]
    cellwise_knots: List[np.ndarray]

    def __init__(self, topology: SplineTopology, nvis: int = 1):
        self.nodal_knots = [
            util.subdivide_linear(basis.knot_spans(), nvis)
            for basis in topology.bases
        ]

        self.cellwise_knots = [
            ((knots := np.array(basis.knot_spans()))[:-1] + knots[1:]) / 2
            for basis in topology.bases
        ]

    def tesselate_topology(self, topology: SplineTopology) -> StructuredTopology:
        celltype = [CellType.Line, CellType.Quadrilateral, CellType.Hexahedron][len(self.nodal_knots) - 1]
        cellshape = tuple(len(knots) - 1 for knots in self.nodal_knots)
        return StructuredTopology(cellshape, celltype)

    def tesselate_field(self, topology: SplineTopology, field: Field, field_data: FieldData) -> FieldData:
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

        coeffs = splipy.utils.reshape(coeffs, shape, order='F')
        new_spline = SplineObject(bases, coeffs, rational=rational, raw=True)
        return FieldData(data=util.flatten_2d(new_spline(*knots)))
