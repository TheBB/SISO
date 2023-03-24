from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import IO, Tuple, TypeVar, Union, cast

from numpy import number
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
from ..api import Basis, Source, Step, Zone
from ..topology import CellType, DiscreteTopology, StructuredTopology
from ..util import FieldData
from .api import Field, OutputMode, Writer, WriterProperties, WriterSettings


class Behavior(Enum):
    OnlyStructured = auto()
    OnlyUnstructured = auto()
    Whatever = auto()


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
T = TypeVar("T", bound=Step)
Z = TypeVar("Z", bound=Zone)
S = TypeVar("S", bound=number)

BackendWriter = Union[vtkXMLWriter, vtkDataWriter]


def transpose(data: FieldData[S], grid: vtkPointSet, cellwise: bool = False) -> FieldData[S]:
    if not isinstance(grid, vtkStructuredGrid):
        return data
    shape = grid.GetDimensions()
    if cellwise:
        i, j, k = shape
        shape = (max(i - 1, 1), max(j - 1, 1), max(k - 1, 1))
    return data.transpose(shape, (2, 1, 0))


def get_grid(
    topology: DiscreteTopology, legacy: bool, behavior: Behavior
) -> Tuple[vtkPointSet, BackendWriter]:
    if isinstance(topology, StructuredTopology) and behavior != Behavior.OnlyUnstructured:
        sgrid = vtkStructuredGrid()
        shape = topology.cellshape
        while len(shape) < 3:
            shape = (*shape, 0)
        sgrid.SetDimensions(*(s + 1 for s in shape))
        if legacy:
            return sgrid, vtkStructuredGridWriter()
        else:
            return sgrid, vtkXMLStructuredGridWriter()

    assert behavior != Behavior.OnlyStructured
    assert topology.celltype in (CellType.Line, CellType.Quadrilateral, CellType.Hexahedron)

    ugrid = vtkUnstructuredGrid()
    cells = (
        FieldData.concat(
            topology.cells.constant_like(topology.cells.num_comps, ncomps=1, dtype=int),
            topology.cells,
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


class VtkWriterBase(ABC, Writer):
    filename: Path
    output_mode: OutputMode = OutputMode.Binary
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
            require_single_basis=True,
            require_discrete_topology=True,
        )

    def configure(self, settings: WriterSettings) -> None:
        if settings.output_mode is not None:
            assert settings.output_mode in (OutputMode.Binary, OutputMode.Ascii)
            self.output_mode = settings.output_mode

    @abstractmethod
    def grid_and_writer(self, topology: DiscreteTopology) -> Tuple[vtkPointSet, BackendWriter]:
        ...

    def consume_timestep(self, timestep: T, filename: Path, source: Source[B, F, T, Z], geometry: F) -> None:
        zone = next(source.zones())
        topology = cast(DiscreteTopology, source.topology(timestep, source.basis_of(geometry), zone))

        grid, writer = self.grid_and_writer(topology)
        apply_output_mode(writer, self.output_mode)

        data = source.field_data(timestep, geometry, zone)
        data = transpose(data, grid, geometry.cellwise)

        points = vtkPoints()
        p = data.ensure_ncomps(3, allow_scalar=False)
        points.SetData(p.vtk())
        grid.SetPoints(points)

        basis = next(source.bases())
        for field in source.fields(basis):
            if field.is_geometry:
                continue
            target = grid.GetCellData() if field.cellwise else grid.GetPointData()
            data = source.field_data(timestep, field, zone)
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

    def consume(self, source: Source[B, F, T, Z], geometry: F) -> None:
        filenames = util.filename_generator(self.filename, source.properties.instantaneous)
        for timestep, filename in zip(source.steps(), filenames):
            self.consume_timestep(timestep, filename, source, geometry)


class VtkWriter(VtkWriterBase):
    allow_nan_in_ascii = False

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> Tuple[vtkPointSet, BackendWriter]:
        return get_grid(topology, legacy=True, behavior=Behavior.Whatever)


class VtuWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> Tuple[vtkPointSet, BackendWriter]:
        return get_grid(topology, legacy=False, behavior=Behavior.OnlyUnstructured)


class VtsWriter(VtkWriterBase):
    allow_nan_in_ascii = True

    def __init__(self, filename: Path):
        super().__init__(filename)

    def grid_and_writer(self, topology: DiscreteTopology) -> Tuple[vtkPointSet, BackendWriter]:
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

    def __exit__(self, *args) -> None:
        super().__exit__(*args)
        self.pvd.write("  </Collection>\n")
        self.pvd.write("</VTKFile>\n")
        self.pvd.__exit__(*args)
        logging.info(self.pvd_filename)

    def consume_timestep(self, timestep: T, filename: Path, source: Source[B, F, T, Z], geometry: F) -> None:
        super().consume_timestep(timestep, filename, source, geometry)
        relative_filename = filename.relative_to(self.pvd_filename.parent)
        if timestep.value is not None:
            time = timestep.value
        else:
            time = timestep.index
        self.pvd.write(f'    <DataSet timestep="{time}" part="0" file="{relative_filename}" />\n')
