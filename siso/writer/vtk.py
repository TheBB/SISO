from __future__ import annotations

import logging
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
from typing_extensions import Self

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

from .. import util
from ..api import FieldType, Source, TimeStep
from ..topology import CellType, DiscreteTopology, StructuredTopology, UnstructuredTopology
from ..util import FieldData
from ..zone import Zone
from .api import Field, OutputMode, Writer, WriterProperties, WriterSettings


class Behavior(Enum):
    OnlyStructured = auto()
    OnlyUnstructured = auto()
    Whatever = auto()


def transpose(data: FieldData, grid: vtkPointSet, cellwise: bool = False):
    if not isinstance(grid, vtkStructuredGrid):
        return data
    shape = grid.GetDimensions()
    if cellwise:
        i, j, k = shape
        shape = (max(i - 1, 1), max(j - 1, 1), max(k - 1, 1))
    return data.transpose(shape, (2, 1, 0))


def get_grid(
    topology: DiscreteTopology, legacy: bool, behavior: Behavior
) -> Tuple[vtkPointSet, Union[vtkXMLWriter, vtkDataWriter]]:
    if isinstance(topology, StructuredTopology) and behavior != Behavior.OnlyUnstructured:
        sgrid = vtkStructuredGrid()
        shape = topology.cellshape
        while len(shape) < 3:
            shape = (*shape, 0)
        # shape = shape[::-1]
        sgrid.SetDimensions(*(s + 1 for s in shape))
        if legacy:
            return sgrid, vtkStructuredGridWriter()
        else:
            return sgrid, vtkXMLStructuredGridWriter()

    assert behavior != Behavior.OnlyStructured
    assert topology.celltype in (CellType.Line, CellType.Quadrilateral, CellType.Hexahedron)

    ugrid = vtkUnstructuredGrid()
    cells = topology.cells
    cells = np.hstack((cells.shape[-1] * np.ones((len(cells), 1), dtype=int), cells)).ravel().astype("i8")
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
    else:
        return ugrid, vtkXMLUnstructuredGridWriter()


def apply_output_mode(writer: Union[vtkXMLWriter, vtkDataWriter], mode: OutputMode) -> None:
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


F = TypeVar("F", bound=Field)
T = TypeVar("T", bound=TimeStep)
Z = TypeVar("Z", bound=Zone)


class VtkWriterBase(Writer):
    filename: Path
    output_mode: OutputMode = OutputMode.Binary
    grid_getter: Callable[[DiscreteTopology], Tuple[vtkPointSet, Union[vtkXMLWriter, vtkDataWriter]]]
    allow_nan_in_ascii: bool

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return

    @property
    def properties(self) -> WriterProperties:
        return WriterProperties(
            require_single_zone=True,
            require_tesselated=True,
        )

    def configure(self, settings: WriterSettings):
        if settings.output_mode is not None:
            assert settings.output_mode in (OutputMode.Binary, OutputMode.Ascii)
            self.output_mode = settings.output_mode

    def consume(self, source: Source[F, T, Z], geometry: F, fields: Sequence[F]):
        filenames = util.filename_generator(self.filename, source.properties.instantaneous)
        for timestep, filename in zip(source.timesteps(), filenames):
            zone = next(source.zones())
            topology = cast(DiscreteTopology, source.topology(timestep, geometry, zone))

            grid, writer = self.grid_getter(topology)
            apply_output_mode(writer, self.output_mode)

            data = source.field_data(timestep, geometry, zone)
            data = transpose(data, grid, geometry.cellwise)

            points = vtkPoints()
            p = data.ensure_ncomps(3, allow_scalar=False)
            points.SetData(p.vtk())
            grid.SetPoints(points)

            for field in fields:
                target = grid.GetCellData() if field.cellwise else grid.GetPointData()
                data = source.field_data(timestep, field, zone)
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


class VtkWriter(VtkWriterBase):
    allow_nan_in_ascii = False

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.grid_getter = partial(get_grid, legacy=True, behavior=Behavior.Whatever)


class VtuWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.grid_getter = partial(get_grid, legacy=False, behavior=Behavior.OnlyUnstructured)


class VtsWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.grid_getter = partial(get_grid, legacy=False, behavior=Behavior.OnlyStructured)
