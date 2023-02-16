from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .api import Writer, WriterProperties, WriterSettings, OutputMode
from ..api import Source, TimeStep
from ..field import Field
from ..topology import CellType, DiscreteTopology, StructuredTopology, UnstructuredTopology
from ..zone import Zone
from .. import util

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkDataSet,
    vtkUnstructuredGrid,
    vtkStructuredGrid,
    VTK_LINE,
    VTK_QUAD,
    VTK_HEXAHEDRON,
)
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter, vtkStructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter, vtkXMLStructuredGridWriter
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray

from typing import cast, Sequence, TypeVar


def get_grid(legacy: bool, topology: DiscreteTopology):
    if isinstance(topology, StructuredTopology):
        sgrid = vtkStructuredGrid()
        shape = topology.cellshape
        while len(shape) < 1:
            shape = (*shape, 0)
        sgrid.SetDimensions(*(s + 1 for s in shape))
        if legacy:
            return sgrid, vtkStructuredGridWriter()
        else:
            return sgrid, vtkXMLStructuredGridWriter()

    assert isinstance(topology, UnstructuredTopology)
    assert topology.celltype in (CellType.Line, CellType.Quadrilateral, CellType.Hexahedron)

    ugrid = vtkUnstructuredGrid()
    cells = topology.cells
    cells = np.hstack((
        cells.shape[-1] * np.ones((len(cells), 1), dtype=int),
        cells
    )).ravel().astype('i8')
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


F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)
Z = TypeVar('Z', bound=Zone)

class VtkWriter(Writer):
    filename: Path
    output_mode: OutputMode = OutputMode.Binary

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self) -> VtkWriter:
        return self

    def __exit__(self, *args):
        ...

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

            grid, writer = get_grid(legacy=True, topology=topology)

            if self.output_mode == OutputMode.Binary:
                writer.SetFileTypeToBinary()
            elif self.output_mode == OutputMode.Ascii:
                writer.SetFileTypeToASCII()

            data = source.field_data(timestep, geometry, zone)

            points = vtkPoints()
            p = data.ensure_ncomps(3, allow_scalar=False)
            print(p.data.shape)
            points.SetData(p.vtk())
            grid.SetPoints(points)

            for field in fields:
                target = grid.GetCellData() if field.cellwise else grid.GetPointData()
                array = source.field_data(timestep, field, zone).vtk()
                array.SetName(field.name)
                target.AddArray(array)

            writer.SetFileName(str(filename))
            writer.SetInputData(grid)
            writer.Write()

            logging.info(filename)
