from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import IO, TYPE_CHECKING, Self

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_QUAD,
    VTK_TRIANGLE,
    vtkCellArray,
    vtkPointSet,
    vtkStructuredGrid,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkIOLegacy import vtkDataWriter, vtkStructuredGridWriter, vtkUnstructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter, vtkXMLUnstructuredGridWriter, vtkXMLWriter

from siso import api, util
from siso.api import Basis, CellOrdering, DiscreteTopology, Field, NodeShape, Source, Step, Topology, Zone
from siso.topology import CellType, StructuredTopology
from siso.util import FieldData

from .api import OutputMode, Writer, WriterProperties, WriterSettings

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from siso.util.field_data import FloatFieldData


class Behavior(Enum):
    OnlyStructured = auto()
    OnlyUnstructured = auto()
    Whatever = auto()


BackendWriter = vtkXMLWriter | vtkDataWriter


def transpose(data: FloatFieldData, grid: vtkPointSet, cellwise: bool = False) -> FloatFieldData:
    if not isinstance(grid, vtkStructuredGrid):
        return data
    shape = [0, 0, 0]
    grid.GetDimensions(shape)
    if cellwise:
        i, j, k = shape
        shape = [max(i - 1, 1), max(j - 1, 1), max(k - 1, 1)]
    return data.transpose(NodeShape(shape), (2, 1, 0))


def get_grid_and_writer[B: Basis, F: Field, S: Step, Z: Zone](
    source: Source[B, F, S, DiscreteTopology, Z], geometry: F, step: S, legacy: bool, behavior: Behavior
) -> tuple[vtkPointSet, BackendWriter]:
    zones = list(source.zones())

    # Special case: structured grids must consist of one zone, and that zone must be structured
    if len(zones) == 1:
        topology = source.topology(step, source.basis_of(geometry), zones[0])

        if isinstance(topology, StructuredTopology) and behavior != Behavior.OnlyUnstructured:
            sgrid = vtkStructuredGrid()
            shape = tuple(topology.cellshape)
            while len(shape) < 3:
                shape = (*shape, 0)
            sgrid.SetDimensions(*(s + 1 for s in shape))
            if legacy:
                return sgrid, vtkStructuredGridWriter()
            return sgrid, vtkXMLStructuredGridWriter()

    if behavior == Behavior.OnlyStructured:
        raise api.Unexpected("Unstructured or multi-zone topology passed to structured-only context")

    ugrid = vtkUnstructuredGrid()
    cells = np.empty((0,), dtype=np.int64)
    celltypes = np.empty((0,), dtype=np.uint8)
    total_nodes = 0
    total_cells = 0

    for zone in zones:
        topology = source.topology(step, source.basis_of(geometry), zone)

        if topology.celltype not in (
            CellType.Line,
            CellType.Triangle,
            CellType.Quadrilateral,
            CellType.Hexahedron,
        ):
            raise api.Unsupported("VTK writer only supports lines, quadrilaterals and hexahedra")

        celltype = {
            CellType.Line: VTK_LINE,
            CellType.Triangle: VTK_TRIANGLE,
            CellType.Quadrilateral: VTK_QUAD,
            CellType.Hexahedron: VTK_HEXAHEDRON,
        }[topology.celltype]
        celltypes = np.hstack([celltypes, np.full((topology.num_cells,), celltype, dtype=np.uint8)])

        cells = np.hstack(
            [
                cells,
                FieldData.join_comps(
                    topology.cells.constant_like(topology.cells.num_comps, ncomps=1, dtype=np.int32),
                    topology.cells_as(CellOrdering.Vtk) + total_nodes,
                )
                .numpy()
                .ravel()
                .astype("i8"),
            ]
        )

        total_nodes += topology.num_nodes
        total_cells += topology.num_cells

    cellarray = vtkCellArray()
    cellarray.SetCells(total_cells, numpy_to_vtkIdTypeArray(cells))

    cell_types_vtk = numpy_to_vtk(celltypes, array_type=VTK_UNSIGNED_CHAR, deep=1)

    ugrid.SetCells(cell_types_vtk, cellarray)

    if legacy:
        return ugrid, vtkUnstructuredGridWriter()
    return ugrid, vtkXMLUnstructuredGridWriter()


def apply_output_mode(writer: vtkXMLWriter | vtkDataWriter, mode: OutputMode) -> None:
    if isinstance(writer, vtkDataWriter):
        if mode == OutputMode.Binary:
            writer.SetFileTypeToBinary()
        elif mode == OutputMode.Ascii:
            writer.SetFileTypeToASCII()
    elif isinstance(writer, vtkXMLWriter):
        if mode == OutputMode.Binary:
            writer.SetDataModeToBinary()
        elif mode == OutputMode.Ascii:
            writer.SetDataModeToAscii()
        elif mode == OutputMode.Appended:
            writer.SetDataModeToAppended()


class VtkWriterBase(ABC, Writer):
    filename: Path
    output_mode: OutputMode = OutputMode.Binary
    allow_nan_in_ascii: bool

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return

    @property
    def properties(self) -> WriterProperties:
        return WriterProperties(
            require_single_basis=True,
            require_discrete_topology=True,
        )

    def configure(self, settings: WriterSettings) -> None:
        if settings.output_mode is not None:
            if settings.output_mode not in (OutputMode.Binary, OutputMode.Ascii):
                raise api.Unsupported(f"Unsupported output mode for VTK: {settings.output_mode}")
            self.output_mode = settings.output_mode

    @abstractmethod
    def grid_and_writer[B: Basis, F: Field, S: Step, Z: Zone](
        self, source: Source[B, F, S, DiscreteTopology, Z], step: S, geometry: F
    ) -> tuple[vtkPointSet, BackendWriter]: ...

    def consume_timestep[B: Basis, F: Field, S: Step, Z: Zone](
        self, step: S, filename: Path, source: Source[B, F, S, DiscreteTopology, Z], geometry: F
    ) -> None:
        grid, writer = self.grid_and_writer(source, step, geometry)
        apply_output_mode(writer, self.output_mode)

        data: FloatFieldData = FieldData(np.empty((0, geometry.num_comps), dtype=np.float64))
        for zone in source.zones():
            new_data = source.field_data(step, geometry, zone)
            data = FieldData.join_dofs(data, new_data)

        data = transpose(data, grid, geometry.cellwise)
        points = vtkPoints()
        points.SetData(data.ensure_ncomps(3, allow_scalar=False).vtk())
        grid.SetPoints(points)

        for field in source.fields(source.single_basis()):
            if field.is_geometry:
                continue
            target = grid.GetCellData() if field.cellwise else grid.GetPointData()

            data = FieldData(np.empty((0, field.num_comps), dtype=np.float64))
            for zone in source.zones():
                new_data = source.field_data(step, field, zone)
                data = FieldData.join_dofs(data, new_data)

            if field.is_displacement:
                data = data.ensure_ncomps(3, allow_scalar=False, pad_right=False)
            else:
                data = data.ensure_ncomps(3, allow_scalar=field.is_scalar)
            data = transpose(data, grid, field.cellwise)

            if self.output_mode == OutputMode.Ascii and not self.allow_nan_in_ascii:
                data = data.nan_filter()

            array = data.vtk()
            array.SetName(field.name)
            target.AddArray(array)

        writer.SetFileName(str(filename))
        writer.SetInputData(grid)
        writer.Write()

        logging.info(filename)

    def consume[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
        self, source: Source[B, F, S, T, Z], geometry: F
    ) -> None:
        casted = source.cast_discrete_topology()
        filenames = util.filename_generator(self.filename, source.properties.instantaneous)
        for step, filename in zip(casted.steps(), filenames):
            self.consume_timestep(step, filename, casted, geometry)


class VtkWriter(VtkWriterBase):
    allow_nan_in_ascii = False

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer[B: Basis, F: Field, S: Step, Z: Zone](
        self, source: Source[B, F, S, DiscreteTopology, Z], step: S, geometry: F
    ) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid_and_writer(source, geometry, step, legacy=True, behavior=Behavior.Whatever)


class VtuWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer[B: Basis, F: Field, S: Step, Z: Zone](
        self, source: Source[B, F, S, DiscreteTopology, Z], step: S, geometry: F
    ) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid_and_writer(source, geometry, step, legacy=False, behavior=Behavior.OnlyUnstructured)


class VtsWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer[B: Basis, F: Field, S: Step, Z: Zone](
        self, source: Source[B, F, S, DiscreteTopology, Z], step: S, geometry: F
    ) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid_and_writer(source, geometry, step, legacy=False, behavior=Behavior.OnlyStructured)


class PvdWriter(VtuWriter):
    pvd_dirname: Path
    pvd_filename: Path
    pvd: IO[str]

    def __init__(self, filename: Path):
        self.pvd_filename = filename
        self.pvd_dirname = filename.with_suffix(f"{filename.suffix}-data")
        super().__init__(self.pvd_dirname / "data.vtu")

    def __enter__(self) -> Self:
        self.pvd_dirname.mkdir(exist_ok=True, parents=True)
        self.pvd = self.pvd_filename.open("w").__enter__()
        self.pvd.write('<VTKFile type="Collection">\n')
        self.pvd.write("  <Collection>\n")
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        self.pvd.write("  </Collection>\n")
        self.pvd.write("</VTKFile>\n")
        self.pvd.__exit__(exc_type, exc_val, exc_tb)
        logging.info(self.pvd_filename)

    def consume_timestep[B: Basis, F: Field, S: Step, Z: Zone](
        self, timestep: S, filename: Path, source: Source[B, F, S, DiscreteTopology, Z], geometry: F
    ) -> None:
        super().consume_timestep(timestep, filename, source, geometry)
        relative_filename = filename.relative_to(self.pvd_filename.parent)
        time = timestep.value if timestep.value is not None else timestep.index
        self.pvd.write(f'    <DataSet timestep="{time}" part="0" file="{relative_filename}" />\n')
