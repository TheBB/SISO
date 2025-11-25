from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import IO, TYPE_CHECKING, Self, TypeVar

from numpy import number
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_QUAD,
    vtkCellArray,
    vtkPointSet,
    vtkStructuredGrid,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkIOLegacy import vtkDataWriter, vtkStructuredGridWriter, vtkUnstructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter, vtkXMLUnstructuredGridWriter, vtkXMLWriter

from siso import api, util
from siso.api import B, CellOrdering, DiscreteTopology, F, NodeShape, S, Source, T, Z
from siso.topology import CellType, StructuredTopology
from siso.util import FieldData

from .api import OutputMode, Writer, WriterProperties, WriterSettings

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class Behavior(Enum):
    OnlyStructured = auto()
    OnlyUnstructured = auto()
    Whatever = auto()


Sc = TypeVar("Sc", bound=number)
BackendWriter = vtkXMLWriter | vtkDataWriter


def transpose(data: FieldData[Sc], grid: vtkPointSet, cellwise: bool = False) -> FieldData[Sc]:
    if not isinstance(grid, vtkStructuredGrid):
        return data
    shape = [0, 0, 0]
    grid.GetDimensions(shape)
    if cellwise:
        i, j, k = shape
        shape = [max(i - 1, 1), max(j - 1, 1), max(k - 1, 1)]
    return data.transpose(NodeShape(shape), (2, 1, 0))


def get_grid(
    topology: DiscreteTopology, legacy: bool, behavior: Behavior
) -> tuple[vtkPointSet, BackendWriter]:
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
        raise api.Unexpected("Unstructured topology passed to structured-only context")
    if topology.celltype not in (CellType.Line, CellType.Quadrilateral, CellType.Hexahedron):
        raise api.Unsupported("VTK writer only supports lines, quadrilaterals and hexahedra")

    ugrid = vtkUnstructuredGrid()
    cells = (
        FieldData.join_comps(
            topology.cells.constant_like(topology.cells.num_comps, ncomps=1, dtype=int),
            topology.cells_as(CellOrdering.Vtk),
        )
        .numpy()
        .ravel()
        .astype("i8")
    )
    cellarray = vtkCellArray()
    cellarray.SetCells(len(cells), numpy_to_vtkIdTypeArray(cells))
    celltype = {
        CellType.Line: VTK_LINE,
        CellType.Quadrilateral: VTK_QUAD,
        CellType.Hexahedron: VTK_HEXAHEDRON,
    }[topology.celltype]
    ugrid.SetCells(celltype, cellarray)

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
            require_single_zone=True,
            require_single_basis=True,
            require_discrete_topology=True,
        )

    def configure(self, settings: WriterSettings) -> None:
        if settings.output_mode is not None:
            if settings.output_mode not in (OutputMode.Binary, OutputMode.Ascii):
                raise api.Unsupported(f"Unsupported output mode for VTK: {settings.output_mode}")
            self.output_mode = settings.output_mode

    @abstractmethod
    def grid_and_writer(self, topology: DiscreteTopology) -> tuple[vtkPointSet, BackendWriter]: ...

    def consume_timestep(
        self, step: S, filename: Path, source: Source[B, F, S, DiscreteTopology, Z], geometry: F
    ) -> None:
        zone = source.single_zone()
        topology = source.topology(step, source.basis_of(geometry), zone)

        grid, writer = self.grid_and_writer(topology)
        apply_output_mode(writer, self.output_mode)

        data = source.field_data(step, geometry, zone)
        data = transpose(data, grid, geometry.cellwise)

        points = vtkPoints()
        p = data.ensure_ncomps(3, allow_scalar=False)
        points.SetData(p.vtk())
        grid.SetPoints(points)

        for field in source.fields(source.single_basis()):
            if field.is_geometry:
                continue
            target = grid.GetCellData() if field.cellwise else grid.GetPointData()
            data = source.field_data(step, field, zone)
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

    def consume(self, source: Source[B, F, S, T, Z], geometry: F) -> None:
        casted = source.cast_discrete_topology()
        filenames = util.filename_generator(self.filename, source.properties.instantaneous)
        for step, filename in zip(casted.steps(), filenames):
            self.consume_timestep(step, filename, casted, geometry)


class VtkWriter(VtkWriterBase):
    allow_nan_in_ascii = False

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid(topology, legacy=True, behavior=Behavior.Whatever)


class VtuWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid(topology, legacy=False, behavior=Behavior.OnlyUnstructured)


class VtsWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> tuple[vtkPointSet, BackendWriter]:
        return get_grid(topology, legacy=False, behavior=Behavior.OnlyStructured)


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

    def consume_timestep(
        self, timestep: S, filename: Path, source: Source[B, F, S, DiscreteTopology, Z], geometry: F
    ) -> None:
        super().consume_timestep(timestep, filename, source, geometry)
        relative_filename = filename.relative_to(self.pvd_filename.parent)
        time = timestep.value if timestep.value is not None else timestep.index
        self.pvd.write(f'    <DataSet timestep="{time}" part="0" file="{relative_filename}" />\n')
