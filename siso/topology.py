"""This module contains all the implementations of the siso.api.Topology
abstract protocol and the objects required to implement the interface.
"""

from __future__ import annotations

import logging
from functools import partial
from io import BytesIO, StringIO
from typing import IO, Dict, Iterable, Iterator, List, Optional, Tuple, overload

import lrspline as lr
import numpy as np
import splipy.utils
from attrs import define
from numpy import floating, integer
from splipy import BSplineBasis, SplineObject
from splipy.io import G2

from . import api, util
from .api import CellType, Coords, DiscreteTopology, Field, FieldDataFilter, Rationality, Topology
from .util import FieldData


@define
class UnstructuredTopology(DiscreteTopology):
    """The 'lowest common denominator' of topologies, the unstructured discrete
    topology is a collection of cells with a given type, each represented as an
    index into an array of nodes.

    This object does not actually have an array of nodes: that is supplied by
    the field data originating from a geometry field.
    """

    num_nodes: int
    cells: FieldData[integer]
    celltype: CellType

    @staticmethod
    def from_ifem(data: bytes) -> Tuple[Coords, UnstructuredTopology, FieldData[floating]]:
        """Special purpose constructor for parsing an IFEM Lagriangian patch
        with hexahedral cells.

        Returns a sequence of coordinates (in this case, just the first point)
        for constructing zone objects, the topology, as well as the field data
        for the nodal array.
        """
        io = BytesIO(data)

        first_line = next(io)
        assert first_line.startswith(b"# LAGRANGIAN")
        _, _, nodespec, elemspec, typespec = first_line.split()

        # Read number of nodes, cells and cell type
        assert nodespec.startswith(b"nodes=")
        assert elemspec.startswith(b"elements=")
        assert typespec.startswith(b"type=")
        num_nodes = int(nodespec.split(b"=", 1)[1])
        num_cells = int(elemspec.split(b"=", 1)[1])
        celltype = typespec.split(b"=", 1)[1]
        assert celltype == b"hexahedron"

        # Read nodal coordinates
        nodes = np.zeros((num_nodes, 3), dtype=float)
        for i in range(num_nodes):
            nodes[i] = list(map(float, next(io).split()))

        # Read cell indices
        cells = np.zeros((num_cells, 8), dtype=int)
        for i in range(num_cells):
            cells[i] = list(map(int, next(io).split()))

        # IFEM uses a different cell numbering than we do, so renumber
        cell_data = FieldData(cells).swap_components(6, 7).swap_components(2, 3)

        corners = (tuple(nodes[0]),)
        topology = UnstructuredTopology(num_nodes, cell_data, CellType.Hexahedron)
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
        """Join two or more unstructered topologies together.

        This does not perform any actual 'joining': the nodes in each topology
        is considered distinct and the resulting topology will have disjoint
        components.

        Supports an arbitrary number of discrete topology arguments, or a single
        iterable of discrete topologies.
        """
        iterable: Iterable[DiscreteTopology] = other if isinstance(other[0], DiscreteTopology) else other[0]
        num_nodes = 0
        celltype: Optional[CellType] = None

        # Utility function for producing cell arrays. Keeps internal track of
        # the number of nodes seen so far, and adjusts the cell indices
        # accordingly. Crashes if cell types are incompatible.
        def consume() -> Iterable[FieldData[integer]]:
            nonlocal num_nodes, celltype
            for topo in iterable:
                if celltype is None:
                    celltype = topo.celltype
                else:
                    assert celltype == topo.celltype
                yield topo.cells + num_nodes
                num_nodes += topo.num_nodes

        # This runs the consume iterator to the end, which should populate the
        # local variables `num_nodes` and `celltype`.
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
        return self.cells.num_dofs

    def discretize(self, nvis: int) -> Tuple[DiscreteTopology, FieldDataFilter]:
        # Discretizing an unstructured topology is a no-op.
        assert nvis == 1
        return self, lambda field, data: data

    def create_merger(self) -> api.TopologyMerger:
        return UnstructuredTopologyMerger(self.num_cells, self.num_nodes, self.celltype)


@define
class UnstructuredTopologyMerger:
    num_cells: int
    num_nodes: int
    celltype: CellType

    def __call__(self, topology: Topology) -> Tuple[Topology, api.FieldDataFilter]:
        # At the moment we do not support the merging of two incompatible
        # unstructured topologies. Therefore assert (to the best of our
        # abilities) that the topology is identical to the source topology, and
        # return it unchanged.
        assert isinstance(topology, UnstructuredTopology)
        assert topology.num_cells == self.num_cells
        assert topology.num_nodes == self.num_nodes
        assert topology.celltype == self.celltype
        return topology, lambda field, data: data


class StructuredTopology:
    """Structured topologies represent a Cartesian tensor product grid of
    cells."""

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
        """Produce an array of cells on demand."""
        return util.structured_cells(self.cellshape, self.pardim)

    @property
    def nodeshape(self) -> Tuple[int, ...]:
        return tuple(n + 1 for n in self.cellshape)

    def discretize(self, nvis: int) -> Tuple[DiscreteTopology, FieldDataFilter]:
        """Discretizing a structured topology is a no-op."""
        assert nvis == 1
        return self, lambda field, data: data

    def create_merger(self) -> api.TopologyMerger:
        return StructuredTopologyMerger(self.cellshape, self.celltype)

    def transpose(self, axes: Tuple[int, ...]) -> StructuredTopology:
        """Return a new structured topology with transposed axes."""
        assert len(axes) == self.pardim
        return StructuredTopology(
            cellshape=tuple(self.cellshape[i] for i in axes),
            celltype=self.celltype,
        )


@define
class StructuredTopologyMerger:
    cellshape: Tuple[int, ...]
    celltype: CellType

    def __call__(self, topology: Topology) -> Tuple[Topology, api.FieldDataFilter]:
        # At the moment we do not support the merging of two incompatible
        # structured topologies. Therefore assert (to the best of our abilities)
        # that the topology is identical to the source topology, and return it
        # unchanged.
        assert isinstance(topology, StructuredTopology)
        assert topology.cellshape == self.cellshape
        assert topology.celltype == self.celltype
        return topology, lambda field, data: data


class G2Object(G2):
    """Utility wrapper for the Splipy G2 reader that can read from arbitrary
    streams, and not just files.
    """

    def __init__(self, fstream: IO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == "w"
        super().__init__("")

    def __enter__(self) -> G2Object:
        return self


@define
class SplineTopology(Topology):
    """A B-Spline or NURBS topology as represented by Splipy."""

    bases: List[BSplineBasis]
    weights: Optional[np.ndarray]

    @staticmethod
    def from_splineobject(obj: SplineObject) -> Tuple[Coords, SplineTopology, FieldData[floating]]:
        """Construct a spline topology from a Splipy SplineObject.

        Returns a sequence of coordinates (the corners) for constructing zone
        objects, the topology, as well as the field data for the nodal array.
        """
        corners = tuple(tuple(point) for point in obj.corners())

        # If NURBS, extract the weights separately (they are part of the
        # topology, not the field data)
        if obj.rational:
            weights = util.transpose_butlast(obj.controlpoints[..., -1:]).flatten()
            cps = obj.controlpoints[..., :-1]
        else:
            weights = None
            cps = obj.controlpoints

        # Reorder and flatten the control point array
        cps = util.flatten_2d(util.transpose_butlast(cps))

        return (
            corners,
            SplineTopology(obj.bases, weights),
            FieldData(cps),
        )

    @staticmethod
    def from_bytes(data: bytes) -> Iterator[Tuple[Coords, SplineTopology, FieldData[floating]]]:
        """Special constructor parsing bytestring in G2-format."""
        yield from SplineTopology.from_string(data.decode())

    @staticmethod
    def from_string(data: str) -> Iterator[Tuple[Coords, SplineTopology, FieldData[floating]]]:
        """Special constructor for parsing a string in G2-format."""
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

    def discretize(self, nvis: int) -> Tuple[DiscreteTopology, FieldDataFilter]:
        tesselator = SplineTesselator(self, nvis)
        discrete = tesselator.tesselate_topology(self)
        return discrete, lambda field, data: tesselator.tesselate_field(self, field, data)

    def create_merger(self) -> api.TopologyMerger:
        return SplineTesselator(self, nvis=1)


class SplineTesselator(api.TopologyMerger):
    """This class handles merging and discretization of spline topologies.

    If `data` is associated with `topology` then after

    ```
    new_topology = self.tesselate_topology(topology)
    new_data = self.tesselate_field(topology, field, field_data)
    ```

    `new_data` will be compatible with `new_topology`. Here, `self` is an
    instance of SplineTesselator.

    Parameters:
    - topology: the 'master' topology. This will be used to determine the
        parametric evaluation points used to discretize other topologies.
    - nvis: number of subdivions per element
    """

    nodal_knots: List[np.ndarray]
    cellwise_knots: List[np.ndarray]

    def __init__(self, topology: SplineTopology, nvis: int = 1):
        # These are the parametric points that will be used to discretize nodal fields
        self.nodal_knots = [util.subdivide_linear(basis.knot_spans(), nvis) for basis in topology.bases]

        # ...and these will be used for cellwise fields (we take the center of
        # each parameter interval)
        self.cellwise_knots = [
            ((knots := np.array(basis.knot_spans()))[:-1] + knots[1:]) / 2 for basis in topology.bases
        ]

    # This method implements the interface for TopologyMerger
    def __call__(self, topology: Topology) -> Tuple[Topology, api.FieldDataFilter]:
        assert isinstance(topology, SplineTopology)
        discrete = self.tesselate_topology(topology)
        mapper = partial(self.tesselate_field, topology)
        return discrete, mapper

    def tesselate_topology(self, topology: SplineTopology) -> StructuredTopology:
        """Discretize a spline topology by returning a structured topology with
        the appropriate cell type and shape.
        """
        celltype = [CellType.Line, CellType.Quadrilateral, CellType.Hexahedron][len(self.nodal_knots) - 1]
        cellshape = tuple(len(knots) - 1 for knots in self.nodal_knots)
        return StructuredTopology(cellshape, celltype)

    def tesselate_field(
        self,
        topology: SplineTopology,
        field: Field,
        field_data: FieldData[floating],
    ) -> FieldData[floating]:
        """Convert a field data array to make it compatible with the discretized
        topology.
        """

        # Not much shared code between the cellwise and nodal branches, unfortunately.
        # Each branch should establish: bases, shape (of control points), coeffs
        # (control points, possibly together with weights), rational, and knots
        # (the parametric points in which to evaluate).
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

        # Construct a spline object and evaluate it
        coeffs = splipy.utils.reshape(coeffs, shape, order="F")
        new_spline = SplineObject(bases, coeffs, rational=rational, raw=True)
        return FieldData(util.flatten_2d(new_spline(*knots)))


@define
class LrTopology(Topology):
    """A LR-Spline topology as represented by LRSplines (possibly rational)."""

    obj: lr.LRSplineObject
    weights: Optional[np.ndarray]

    @staticmethod
    def from_lrobject(
        obj: lr.LRSplineObject,
        rationality: Optional[Rationality],
    ) -> Tuple[Coords, LrTopology, FieldData[floating]]:
        """Construct an LR topology from an LRSplineObject.

        Returns a sequence of coordinates (the corners) for constructing zone
        objects, the topology, as well as the field data for the nodal array.
        """

        corners = tuple(tuple(point) for point in obj.corners())
        if rationality == Rationality.Always:
            rational = True
        elif rationality == Rationality.Never:
            rational = False
        else:
            # If not explicitly overridden, we interpret objects as being
            # rational if they have more control point components than
            # parametric dimensions. Unfortunately LRSplines don't have explicit
            # support for rationality, so we have to work around it.
            rational = obj.dimension > obj.pardim
            if rational:
                logging.warning(
                    f"Treating LR spline with parametric dimension {obj.pardim} "
                    f"and physical dimension {obj.dimension} as rational"
                )
                logging.warning(
                    "Use --rational/--non-rational to override this behavior " "or suppress this warning"
                )

        # Separate weights and control points (weights belong to the topology)
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
    def from_bytes(
        data: bytes,
        rationality: Optional[Rationality],
    ) -> Iterator[Tuple[Coords, LrTopology, FieldData[floating]]]:
        """Special constructor parsing bytestring in LR-format."""
        yield from LrTopology.from_string(data.decode(), rationality)

    @staticmethod
    def from_string(
        data: str,
        rationality: Optional[Rationality],
    ) -> Iterator[Tuple[Coords, LrTopology, FieldData[floating]]]:
        """Special constructor parsing string in LR-format."""
        for obj in lr.LRSplineObject.read_many(StringIO(data)):
            yield LrTopology.from_lrobject(obj, rationality)

    @property
    def pardim(self) -> int:
        return self.obj.pardim

    @property
    def num_nodes(self) -> int:
        return len(self.obj)

    @property
    def num_cells(self) -> int:
        return len(self.obj.elements)

    def discretize(self, nvis: int) -> Tuple[DiscreteTopology, FieldDataFilter]:
        tesselator = LrTesselator(self.obj, self.weights, nvis)
        discrete = tesselator.tesselate_topology(self)
        return discrete, lambda field, data: tesselator.tesselate_field(self, field, data)

    def create_merger(self) -> api.TopologyMerger:
        return LrTesselator(self.obj, self.weights, nvis=1)


class LrTesselator(api.TopologyMerger):
    """This class handles merging and discretization of LR spline topologies.

    If `data` is associated with `topology` then after

    ```
    new_topology = self.tesselate_topology(topology)
    new_data = self.tesselate_field(topology, field, field_data)
    ```

    `new_data` will be compatible with `new_topology`. Here, `self` is an
    instance of LrTesselator.

    Parameters:
    - obj: the 'master' LRSplineObject. This will be used to determine the
        parametric evaluation points used to discretize other topologies.
    - weights: weight array for rational evaluation
    - nvis: number of subdivions per element
    """

    nodes: np.ndarray
    cells: FieldData[integer]
    weights: Optional[np.ndarray]
    nvis: int

    def __init__(self, obj: lr.LRSplineObject, weights: Optional[np.ndarray], nvis: int):
        # Dictionary mapping parametric values to node index
        nodes: Dict[Tuple[float, ...], int] = {}

        # Cells represented as nodal indices
        cells: List[List[int]] = []

        # The visitor function, when called on each element, will populate the
        # above two data structures.
        visitor = util.visit_face if obj.pardim == 2 else util.visit_volume
        for element in obj.elements:
            visitor(element, nodes, cells, nvis=1)

        self.nodes = FieldData.from_iter(nodes).numpy()
        self.cells = FieldData(np.array(cells, dtype=int))
        self.weights = weights
        self.nvis = nvis

    # This method implements the interface for TopologyMerger
    def __call__(self, topology: Topology) -> Tuple[Topology, api.FieldDataFilter]:
        assert isinstance(topology, LrTopology)
        discrete = self.tesselate_topology(topology)
        mapper = partial(self.tesselate_field, topology)
        return discrete, mapper

    def tesselate_topology(self, topology: LrTopology) -> DiscreteTopology:
        """Discretize an LR-spline topology by returning an unstructured
        topology with the appropriate cell type and shape.
        """
        celltype = CellType.Hexahedron if topology.pardim == 3 else CellType.Quadrilateral
        return UnstructuredTopology(len(self.nodes), self.cells, celltype)

    def tesselate_field(
        self,
        topology: LrTopology,
        field: Field,
        field_data: FieldData[floating],
    ) -> FieldData[floating]:
        """Convert a field data array to make it compatible with the discretized
        topology.
        """
        if field.cellwise:
            # Calculate the centers of each cell by averaging the surrounding nodes
            cell_centers = (np.mean(self.nodes[c], axis=0) for c in self.cells.vectors)

            # Get the LRSpline internal cell ID for each cell center by calling
            # element_at(), and use that to index into the field data array.
            return FieldData.from_iter(
                field_data.numpy()[topology.obj.element_at(*c).id] for c in cell_centers
            )

        else:
            # Clone the LRSpline object and replace its control points with the
            # field we want to evaluate. Then evaluate at the nodal parametric
            # points.
            obj = topology.obj.clone()
            coeffs = field_data.data
            if self.weights is not None:
                coeffs = np.hstack((coeffs, self.weights.reshape(-1, 1)))
            obj.controlpoints = coeffs
            evaluated = FieldData.from_iter(obj(*node) for node in self.nodes)
            if self.weights is not None:
                evaluated = evaluated.collapse_weights()
            return evaluated
