import vtk
import vtk.util.numpy_support as vnp
import numpy as np
from os.path import splitext
from collections import namedtuple, OrderedDict

import treelog as log

from .. import config


Patch = namedtuple('Patch', ['nodes', 'elements', 'dim'])
Field = namedtuple('Field', ['name', 'cells', 'kind', 'results'])


class AbstractVTKWriter:

    def __init__(self, filename):
        self.filename = filename

        self.validate_mode()

    def __enter__(self):
        self.patches = OrderedDict()
        self.fields = OrderedDict()
        self.stepid = 0
        return self

    def __exit__(self, type_, value, backtrace):
        pass

    def make_filename(self):
        if not config.multiple_timesteps:
            return self.filename
        fn, ext = splitext(self.filename)
        return '{}-{}{}'.format(fn, self.stepid, ext)

    def writer(self):
        raise NotImplementedError

    def validate_mode(self):
        raise NotImplementedError

    def nan_filter(self, results):
        I, J = np.where(np.isnan(results))
        if len(I) > 0 and config.output_mode == 'ascii':
            log.warning("VTK ASCII files do not support NaN, will be set to zero")
            results[I, J] = 0.0
        return results

    def add_step(self, **data):
        self.step_data = data
        self.stepid += 1

    def update_geometry(self, nodes, elements, dim, patchid):
        nodes = nodes.reshape(-1, nodes.shape[-1])
        self.patches[patchid] = Patch(nodes, elements, dim)

    def finalize_geometry(self, stepid):
        pass

    def update_field(self, results, name, stepid, patchid, kind='scalar', cells=False):
        field = self.fields.setdefault(name, Field(name, cells, kind, OrderedDict()))
        assert field.cells == cells
        assert field.kind == kind
        if kind == 'scalar':
            results = results.reshape(-1, 1)
        elif kind in ('vector', 'displacement'):
            results = results.reshape(-1, 3)
        else:
            assert False
        results = self.nan_filter(results)
        field.results[patchid] = results

    def finalize_step(self):
        grid = vtk.vtkUnstructuredGrid()

        allpoints = np.vstack([p.nodes for p in self.patches.values()])
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(allpoints))
        grid.SetPoints(points)

        if all(p.dim == 2 for p in self.patches.values()):
            npts = 4
            celltype = vtk.VTK_QUAD
        elif all(p.dim == 3 for p in self.patches.values()):
            npts = 8
            celltype = vtk.VTK_HEXAHEDRON
        else:
            assert False

        offset = np.hstack([[0], np.cumsum([len(p.nodes) for p in self.patches.values()])])
        elements = np.vstack([patch.elements + off for patch, off in zip(self.patches.values(), offset)])
        elements = np.hstack([npts * np.ones((elements.shape[0],1), dtype=int), elements])
        cells = vtk.vtkCellArray()
        cells.SetCells(len(elements), vnp.numpy_to_vtkIdTypeArray(elements.ravel(), deep=True))
        grid.SetCells(celltype, cells)

        pointdata = grid.GetPointData()
        celldata = grid.GetCellData()

        for field in self.fields.values():
            data = np.vstack([k for k in field.results.values()])
            array = vnp.numpy_to_vtk(data)
            array.SetName(field.name)
            if field.cells:
                celldata.AddArray(array)
            else:
                pointdata.AddArray(array)

        filename = self.make_filename()
        writer = self.writer()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()


class Writer(AbstractVTKWriter):

    def validate_mode(self):
        if not config.output_mode in ('ascii', 'binary'):
            raise ValueError("VTK format does not support '{}' mode".format(config.output_mode))

    def writer(self):
        writer = vtk.vtkUnstructuredGridWriter()
        if config.output_mode == 'ascii':
            writer.SetFileTypeToASCII()
        else:
            writer.SetFileTypeToBinary()
        return writer
