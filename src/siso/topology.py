"""This module contains all the implementations of the siso.api.Topology
abstract protocol and the objects required to implement the interface.
"""

from __future__ import annotations

import logging
from functools import partial
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, TextIO, cast, overload

import lrspline as lr
import numpy as np
import splipy.utils
from attrs import define
from splipy import BSplineBasis, SplineObject
from splipy.io import G2

from . import api, util
from .api import (
    CellOrdering,
    CellShape,
    CellType,
    DiscreteTopology,
    Field,
    FieldDataFilter,
    NodeShape,
    Point,
    Points,
    Rationality,
    Topology,
)
from .util import FieldData, cell_numbering

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from siso.types import FloatArray, Int, IntVector, Matrix, f64d, i32d
    from siso.util.field_data import FloatFieldData, IntFieldData


# Map of supported declared celltypes
KNOWN_DECLARED_CELLTYPES: dict[bytes, CellType] = {
    b"hexahedron": CellType.Hexahedron,
    b"line": CellType.Line,
    b"quad": CellType.Quadrilateral,
}

# Maps declared celltype + number of entries to actual celltype + degree
KNOWN_CELLTYPES: dict[tuple[CellType, int], tuple[CellType, int] | None] = {
    (CellType.Line, 2): (CellType.Line, 1),
    (CellType.Hexahedron, 8): (CellType.Hexahedron, 1),
    (CellType.Hexahedron, 27): (CellType.Hexahedron, 2),
    (CellType.Quadrilateral, 1): None,
    (CellType.Quadrilateral, 3): (CellType.Triangle, 1),
    (CellType.Quadrilateral, 4): (CellType.Quadrilateral, 1),
}


class DiscreteTopologyImpl(DiscreteTopology):
    def cells_as(self, ordering: CellOrdering) -> IntFieldData:
        permutation = cell_numbering.permute_to(self.celltype, self.degree, ordering)
        return self.cells.permute_components(permutation)


@define
class UnstructuredTopology(DiscreteTopologyImpl):
    """The 'lowest common denominator' of topologies, the unstructured discrete
    topology is a collection of cells with a given type, each represented as an
    index into an array of nodes.

    This object does not actually have an array of nodes: that is supplied by
    the field data originating from a geometry field.
    """

    num_nodes: int
    cells: IntFieldData
    celltype: CellType
    degree: int

    @staticmethod
    def from_ifem(
        data: bytes,
    ) -> Iterator[
        tuple[
            Points,
            UnstructuredTopology,
            FieldData[f64d],
            IntVector,
            IntVector,
            int,
            int,
        ]
    ]:
        """Special purpose constructor for parsing an IFEM Lagriangian patch
        with hexahedral cells.

        Returns a sequence of coordinates (in this case, just the first point)
        for constructing zone objects, the topology, as well as the field data
        for the nodal array.
        """
        io = BytesIO(data)

        first_line = next(io)
        if not first_line.startswith(b"# LAGRANGIAN"):
            raise api.BadInput("Expected '# LAGRANGIAN'")
        _, _, nodespec, elemspec, typespec = first_line.split()

        # Read number of nodes, cells and cell type
        if not nodespec.startswith(b"nodes="):
            raise api.BadInput("Expected 'nodes='")
        if not elemspec.startswith(b"elements="):
            raise api.BadInput("Expected 'elements='")
        if not typespec.startswith(b"type="):
            raise api.BadInput("Expected 'type='")

        num_nodes = int(nodespec.split(b"=", 1)[1])
        num_cells = int(elemspec.split(b"=", 1)[1])
        celltype_id = typespec.split(b"=", 1)[1]

        if celltype_id not in KNOWN_DECLARED_CELLTYPES:
            raise api.Unsupported(f"IFEM cell type '{celltype_id.decode()}'")
        declared_celltype = KNOWN_DECLARED_CELLTYPES[celltype_id]

        # Read nodal coordinates
        nodes = np.zeros((num_nodes, 3), dtype=np.float64)
        for i in range(num_nodes):
            nodes[i] = list(map(float, next(io).split()))

        # Multiple blocks for different cell types. The cell type in the file
        # can lie: we can't guarantee that the file contains uniform cell types.
        blocks: dict[tuple[CellType, int], tuple[IntFieldData, list[int]]] = {}

        for cellno in range(num_cells):
            line = next(io)
            cell_ids = list(map(int, line.split()))

            # Figure out the actual cell type based on the declared cell type
            # and the number of indices.
            d_celltype = KNOWN_CELLTYPES[declared_celltype, len(cell_ids)]
            if d_celltype is None:
                continue
            actual_celltype, degree = d_celltype

            if (actual_celltype, degree) not in blocks:
                blocks[actual_celltype, degree] = (
                    FieldData(np.empty((0, len(cell_ids)), dtype=np.int32)),
                    [],
                )

            origdata, cellnos = blocks[actual_celltype, degree]
            cellnos.append(cellno)
            origdata = FieldData.join_dofs(origdata, FieldData(np.array([cell_ids], dtype=np.int32)))
            blocks[actual_celltype, degree] = (origdata, cellnos)

        # For each block, yield
        for (celltype, degree), (cell_data, cellnos) in blocks.items():
            permutation = cell_numbering.permute_from(celltype, degree, cell_numbering.CellOrdering.Ifem)
            cell_data = cell_data.permute_components(permutation)

            # Get the unique node IDs that this block references
            unique_node_ids = cast("IntVector", np.unique(cell_data.data))

            # Get the corners of the block
            corners = FieldData(nodes).slice_dofs(unique_node_ids).bounding_corners()

            # Reverse the mapping
            rev_node_ids = np.empty((num_nodes,), dtype=np.int64)
            rev_node_ids[unique_node_ids] = np.arange(len(unique_node_ids))

            # Apply the mapping
            cell_data.data.flat[:] = rev_node_ids[cell_data.data.flat]

            topology = UnstructuredTopology(len(unique_node_ids), cell_data, celltype, degree)
            yield (
                corners,
                topology,
                FieldData(nodes),
                unique_node_ids,
                np.array(cellnos, dtype=np.int32),
                num_nodes,
                num_cells,
            )

    @overload
    @staticmethod
    def join(other: Iterable[DiscreteTopology], /) -> UnstructuredTopology: ...

    @overload
    @staticmethod
    def join(*other: DiscreteTopology) -> UnstructuredTopology: ...

    @staticmethod
    def join(*other) -> UnstructuredTopology:  # type: ignore[no-untyped-def]
        """Join two or more unstructured topologies together.

        This does not perform any actual 'joining': the nodes in each topology
        is considered distinct and the resulting topology will have disjoint
        components.

        Supports an arbitrary number of discrete topology arguments, or a single
        iterable of discrete topologies.
        """
        iterable: Iterable[DiscreteTopology] = other if isinstance(other[0], DiscreteTopology) else other[0]
        num_nodes = 0
        celltype: CellType | None = None
        degree: int | None = None

        # Utility function for producing cell arrays. Keeps internal track of
        # the number of nodes seen so far, and adjusts the cell indices
        # accordingly. Crashes if cell types are incompatible.
        def consume() -> Iterable[IntFieldData]:
            nonlocal num_nodes, celltype, degree
            for topo in iterable:
                if celltype is None:
                    celltype = topo.celltype
                    degree = topo.degree
                elif celltype != topo.celltype or degree != topo.degree:
                    raise api.Unsupported("Joining incompatible unstructured topologoies")
                yield topo.cells + num_nodes
                num_nodes += topo.num_nodes

        # This runs the consume iterator to the end, which should populate the
        # local variables `num_nodes`, `degree` and `celltype`.
        cells = FieldData.join_dofs(consume())

        if not celltype or degree is None:
            raise api.Unexpected("Joining of zero discrete topologies")

        return UnstructuredTopology(
            num_nodes=num_nodes,
            cells=cells,
            celltype=celltype,
            degree=degree,
        )

    @property
    def pardim(self) -> int:
        return {CellType.Line: 1, CellType.Quadrilateral: 2, CellType.Hexahedron: 3}[self.celltype]

    @property
    def num_cells(self) -> int:
        return self.cells.num_dofs

    def discretize(self, nvis: int) -> tuple[DiscreteTopology, FieldDataFilter]:
        # Discretizing an unstructured linear topology with nvis=1 is a no-op
        if nvis == 1 and self.degree == 1:
            return self, lambda field, data: data

        # So far we only support nvis=1 for this operation
        if nvis != 1:
            raise api.Unsupported("discretization of unstructured topologies with nvis > 1")
        tesselator = UnstructuredTesselator(self)
        discrete = tesselator.tesselate_topology(self)
        return discrete, lambda field, data: tesselator.tesselate_field(field, data)

    def create_merger(self) -> api.TopologyMerger:
        def merger(topology: Topology) -> tuple[Topology, api.FieldDataFilter]:
            # At the moment we do not support the merging of two incompatible
            # unstructured topologies. Therefore assert (to the best of our
            # abilities) that the topology is identical to the source topology, and
            # return it unchanged.
            if (
                not isinstance(topology, UnstructuredTopology)
                or topology.num_cells != self.num_cells
                or topology.num_nodes != self.num_nodes
                or topology.celltype != self.celltype
                or topology.degree != self.degree
            ):
                raise api.Unsupported("Merging incompatible unstructured topologies")
            return topology, lambda _, data: data

        return merger


class UnstructuredTesselator:
    """This class handles discretization of unstructured topologies.

    If `data` is associated with `topology` then after

    ```
    new_topology = self.tesselate_topology(topology)
    new_data = self.tesselate_field(topology, field, field_data)
    ```

    `new_data` will be compatible with `new_topology`. Here, `self` is an
    instance of UnstructuredTesselator.

    Parameters:
    - master: the 'master' unstructured topology. This will be used to determine
        the parametric evaluation points used to discretize other topologies.
    """

    master: UnstructuredTopology

    # Which nodal indexes to extract for nodal fields
    pick_indexes: list[int]

    # Cell array of new topology
    new_cells: FieldData[i32d]

    def __init__(self, master: UnstructuredTopology):
        self.master = master
        if not master.celltype.is_tensor:
            raise api.Unsupported("Discretizing unstructured topologies with non-tensor cell type")

        # Numpy indexing expression to extract corners
        index = np.ix_(*([(0, -1)] * master.celltype.pardim))

        # Collection of nodal indices which should be retained
        retain: set[int] = set()

        # New cells, in terms of old node indices
        new_cells: list[list[Int]] = []

        # For each cell, extract its corners, mark them as retained, and create
        # a new cell.
        for cell in master.cells.dofs:
            new_cell = cell.reshape(3, 3, 3)[index].flatten()
            new_cells.append(list(new_cell))
            retain.update(new_cell)

        # Map old node numbers to new ones. Initialize to -1 to catch errors.
        new_node_numbers = -np.ones((master.num_nodes,), dtype=int)

        # The indices pick for nodal fields is the collection of retained nodes,
        # in order of increasing (old) index.
        self.pick_indexes = sorted(retain)

        # Give the retained nodes new node numbers.
        new_node_numbers[self.pick_indexes] = np.arange(len(retain))

        # Extract the new node numbers for the cell array.
        self.new_cells = FieldData(new_node_numbers[np.array(new_cells, dtype=np.int32)])
        if (self.new_cells.data < 0).any():
            raise api.Unexpected("Nodes that shouldn't be picked were picked")

    def tesselate_topology(self, topology: UnstructuredTopology) -> UnstructuredTopology:
        # Assert, to the best of our ability, that the topologies are compatible.
        if (
            topology.num_cells != self.master.num_cells
            or topology.num_nodes != self.master.num_nodes
            or topology.celltype != self.master.celltype
            or topology.degree != self.master.degree
        ):
            raise api.Unsupported("Discretizing incompatible unstructured topologies")

        return UnstructuredTopology(
            len(self.pick_indexes),
            self.new_cells,
            CellType.Hexahedron,
            degree=1,
        )

    def tesselate_field(self, field: Field, data: FloatFieldData) -> FloatFieldData:
        # Cellwise fields: the new topology has the same cells as the master topology.
        if field.cellwise:
            return data
        return data.slice_dofs(self.pick_indexes)
        # return FieldData(data.data[self.pick_indexes, :])


@define
class StructuredTopology(DiscreteTopologyImpl):
    """Structured topologies represent a Cartesian tensor product grid of
    cells."""

    cellshape: CellShape
    celltype: CellType
    degree: int

    @property
    def pardim(self) -> int:
        return self.cellshape.pardim

    @property
    def num_cells(self) -> int:
        return util.prod(self.cellshape)

    @property
    def num_nodes(self) -> int:
        return util.prod(s + 1 for s in self.cellshape)

    @property
    def cells(self) -> IntFieldData:
        """Produce an array of cells on demand."""
        return util.structured_cells(self.cellshape, self.pardim)

    @property
    def nodeshape(self) -> tuple[int, ...]:
        return tuple(n + 1 for n in self.cellshape)

    def discretize(self, nvis: int) -> tuple[DiscreteTopology, FieldDataFilter]:
        """Discretizing a structured topology is a no-op."""
        if nvis != 1:
            raise api.Unsupported("Discretizing structured topologies with nvis > 1")
        if self.degree != 1:
            raise api.Unsupported("Discretizing superlinear structured topologies")
        return self, lambda field, data: data

    def create_merger(self) -> api.TopologyMerger:
        return StructuredTopologyMerger(self.cellshape, self.celltype, self.degree)

    def transpose(self, axes: tuple[int, ...]) -> StructuredTopology:
        """Return a new structured topology with transposed axes."""
        if len(axes) != self.pardim:
            raise api.Unexpected(
                "Transposition of structured topology: number of axes must match parametric dimension"
            )
        return StructuredTopology(
            cellshape=CellShape(tuple(self.cellshape[i] for i in axes)),
            celltype=self.celltype,
            degree=self.degree,
        )


@define
class StructuredTopologyMerger:
    cellshape: CellShape
    celltype: CellType
    degree: int

    def __call__(self, topology: Topology) -> tuple[Topology, api.FieldDataFilter]:
        # At the moment we do not support the merging of two incompatible
        # structured topologies. Therefore assert (to the best of our abilities)
        # that the topology is identical to the source topology, and return it
        # unchanged.
        if (
            not isinstance(topology, StructuredTopology)
            or topology.cellshape != self.cellshape
            or topology.celltype != self.celltype
            or topology.degree != self.degree
        ):
            raise api.Unsupported("Merging incompatible structured topologies")
        return topology, lambda field, data: data


class G2Object(G2):
    """Utility wrapper for the Splipy G2 reader that can read from arbitrary
    streams, and not just files.
    """

    def __init__(self, fstream: TextIO, mode: str):
        self.fstream = fstream
        self.onlywrite = mode == "w"
        super().__init__("")

    def __enter__(self) -> G2Object:
        return self


@define
class SplineTopology(Topology):
    """A B-Spline or NURBS topology as represented by Splipy."""

    bases: list[BSplineBasis]
    weights: FloatArray | None

    @staticmethod
    def from_splineobject(obj: SplineObject) -> tuple[Points, SplineTopology, FloatFieldData]:
        """Construct a spline topology from a Splipy SplineObject.

        Returns a sequence of coordinates (the corners) for constructing zone
        objects, the topology, as well as the field data for the nodal array.
        """
        corners = Points(tuple(Point(tuple(point)) for point in obj.corners()))

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
    def from_bytes(data: bytes) -> Iterator[tuple[Points, SplineTopology, FloatFieldData]]:
        """Special constructor parsing bytestring in G2-format."""
        yield from SplineTopology.from_string(data.decode())

    @staticmethod
    def from_string(data: str) -> Iterator[tuple[Points, SplineTopology, FloatFieldData]]:
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

    def discretize(self, nvis: int) -> tuple[DiscreteTopology, FieldDataFilter]:
        tesselator = SplineTesselator(self, nvis)
        discrete = tesselator.tesselate_topology(self)

        def field_data_filter(field: Field, data: FloatFieldData) -> FloatFieldData:
            return tesselator.tesselate_field(self, field, data)

        return discrete, field_data_filter

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

    nodal_knots: list[np.ndarray]
    cellwise_knots: list[np.ndarray]

    def __init__(self, topology: SplineTopology, nvis: int = 1):
        # These are the parametric points that will be used to discretize nodal fields
        self.nodal_knots = [util.subdivide_linear(basis.knot_spans(), nvis) for basis in topology.bases]

        # ...and these will be used for cellwise fields (we take the center of
        # each parameter interval)
        self.cellwise_knots = [
            np.repeat(
                ((knots := np.array(basis.knot_spans()))[:-1] + knots[1:]) / 2,
                nvis,
            )
            for basis in topology.bases
        ]

    # This method implements the interface for TopologyMerger
    def __call__(self, topology: Topology) -> tuple[Topology, api.FieldDataFilter]:
        if not isinstance(topology, SplineTopology):
            raise api.Unsupported("Merging non-spline topology with spline topology")
        discrete = self.tesselate_topology(topology)
        mapper = partial(self.tesselate_field, topology)
        return discrete, mapper

    def tesselate_topology(self, topology: SplineTopology) -> StructuredTopology:
        """Discretize a spline topology by returning a structured topology with
        the appropriate cell type and shape.
        """
        celltype = [CellType.Line, CellType.Quadrilateral, CellType.Hexahedron][len(self.nodal_knots) - 1]
        cellshape = NodeShape(tuple(len(knots) for knots in self.nodal_knots)).cellular
        return StructuredTopology(cellshape, celltype, degree=1)

    def tesselate_field(
        self,
        topology: SplineTopology,
        field: Field,
        field_data: FloatFieldData,
    ) -> FloatFieldData:
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
            # coeffs = field_data.data
            rational = False

        else:
            bases = topology.bases
            shape = tuple(basis.num_functions() for basis in topology.bases)
            knots = self.nodal_knots
            # coeffs = field_data.data
            if topology.weights is not None:
                field_data = FieldData.join_comps(field_data, topology.weights.flatten())
                # coeffs = np.hstack((coeffs, util.flatten_2d(topology.weights)), dtype=coeffs.dtype)
            rational = topology.weights is not None

        # Construct a spline object and evaluate it
        coeffs = splipy.utils.reshape(field_data.data, shape, order="F")
        new_spline = SplineObject(bases, coeffs, rational=rational, raw=True)
        return FieldData(util.flatten_2d(new_spline(*knots)))


@define
class LrTopology(Topology):
    """A LR-Spline topology as represented by LRSplines (possibly rational)."""

    obj: lr.LRSplineObject
    weights: np.ndarray | None

    @staticmethod
    def from_lrobject(
        obj: lr.LRSplineObject,
        rationality: Rationality | None,
    ) -> tuple[Points, LrTopology, FieldData[f64d]]:
        """Construct an LR topology from an LRSplineObject.

        Returns a sequence of coordinates (the corners) for constructing zone
        objects, the topology, as well as the field data for the nodal array.
        """

        corners = Points(tuple(Point(tuple(point)) for point in obj.corners()))

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
                    "Use --rational/--non-rational to override this behavior or suppress this warning"
                )

        # Separate weights and control points (weights belong to the topology)
        if rational:
            weights = obj.controlpoints[:, -1]
            cps = cast("Matrix[f64d]", obj.controlpoints[:, :-1])
        else:
            weights = None
            cps = cast("Matrix[f64d]", obj.controlpoints)

        return (
            corners,
            LrTopology(obj=obj, weights=weights),
            FieldData(cps),
        )

    @staticmethod
    def from_bytes(
        data: bytes,
        rationality: Rationality | None,
    ) -> Iterator[tuple[Points, LrTopology, FieldData[f64d]]]:
        """Special constructor parsing bytestring in LR-format."""
        yield from LrTopology.from_string(data.decode(), rationality)

    @staticmethod
    def from_string(
        data: str,
        rationality: Rationality | None,
    ) -> Iterator[tuple[Points, LrTopology, FieldData[f64d]]]:
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

    def discretize(self, nvis: int) -> tuple[DiscreteTopology, FieldDataFilter]:
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
    cells: IntFieldData
    weights: np.ndarray | None
    nvis: int

    def __init__(self, obj: lr.LRSplineObject, weights: np.ndarray | None, nvis: int):
        # Dictionary mapping parametric values to node index
        nodes: dict[tuple[float, ...], int] = {}

        # Cells represented as nodal indices
        cells: list[list[int]] = []

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
    def __call__(self, topology: Topology) -> tuple[Topology, api.FieldDataFilter]:
        if not isinstance(topology, LrTopology):
            raise api.Unsupported("Merging non-LR topology with LR topology")
        discrete = self.tesselate_topology(topology)
        mapper = partial(self.tesselate_field, topology)
        return discrete, mapper

    def tesselate_topology(self, topology: LrTopology) -> DiscreteTopology:
        """Discretize an LR-spline topology by returning an unstructured
        topology with the appropriate cell type and shape.
        """
        celltype = CellType.Hexahedron if topology.pardim == 3 else CellType.Quadrilateral
        return UnstructuredTopology(len(self.nodes), self.cells, celltype, degree=1)

    def tesselate_field(
        self,
        topology: LrTopology,
        field: Field,
        field_data: FloatFieldData,
    ) -> FloatFieldData:
        """Convert a field data array to make it compatible with the discretized
        topology.
        """
        if field.cellwise:
            # Calculate the centers of each cell by averaging the surrounding nodes
            cell_centers = (np.mean(self.nodes[c], axis=0) for c in self.cells.dofs)

            # Get the LRSpline internal cell ID for each cell center by calling
            # element_at(), and use that to index into the field data array.
            return FieldData.from_iter(
                field_data.numpy()[topology.obj.element_at(*c).id] for c in cell_centers
            )

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
