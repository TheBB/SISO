from pathlib import Path

import netCDF4
from numpy import newaxis as __
import numpy as np

from typing import Optional

from .. import config
from .reader import Reader
from ..writer import Writer
from ..geometry import Quad, Hex, StructuredPatch
from ..fields import SimpleFieldPatch
from ..util import unstagger


MEAN_EARTH_RADIUS = 6_371_000


class WRFReader(Reader):

    reader_name = "NETCDF4-WRF"

    filename: Path
    nc: netCDF4.Dataset

    @classmethod
    def applicable(cls, filepath: Path) -> bool:
        try:
            with netCDF4.Dataset(filepath, 'r') as f:
                assert 'WRF' in f.TITLE
            return True
        except:
            return False

    def __init__(self, filename: Path):
        self.filename = filename

        # Disable periodicity except in global mapping
        if config.mapping == 'local':
            config.require(periodic=False)

    def __enter__(self):
        self.nc = netCDF4.Dataset(self.filename, 'r').__enter__()
        return self

    def __exit__(self, *args):
        self.nc.__exit__(*args)

    @property
    def nsteps(self) -> int:
        return len(self.nc.dimensions['Time'])

    @property
    def nvertical(self) -> int:
        return len(self.nc.dimensions['bottom_top'])

    def variable_at(self, name: str, stepid: int, extrude_if_planar: bool = False) -> np.ndarray:
        time, *dimensions = self.nc[name].dimensions
        assert time == 'Time'
        data = self.nc[name][stepid, ...]

        # Detect staggered axes and un-stagger them
        for i, dim in enumerate(dimensions):
            if dim.endswith('_stag'):
                data = unstagger(data, i)

        # Double-up the periodic axis
        if config.periodic and dimensions[-1].startswith('west_east'):
            data = np.append(data, data[..., :1], axis=-1)

        # Extrude the vertical direction if desired
        if extrude_if_planar and self.variable_type(name) == 'planar':
            newdata = np.zeros((self.nvertical,) + data.shape, dtype=data.dtype)
            newdata[...] = data
            data = newdata

        return data

    def variable_type(self, name: str) -> Optional[str]:
        """Check if a variable is volumetric, planar or none of the above."""
        time, *dimensions = self.nc[name].dimensions
        if time != 'Time':
            return None

        try:
            assert len(dimensions) == 2
            assert dimensions[0].startswith('south_north')
            assert dimensions[1].startswith('west_east')
            return 'planar'
        except AssertionError:
            pass

        try:
            assert len(dimensions) == 3
            assert dimensions[0].startswith('bottom_top')
            assert dimensions[1].startswith('south_north')
            assert dimensions[2].startswith('west_east')
            return 'volumetric'
        except AssertionError:
            pass

        return None

    def patch_at(self, stepid: int) -> StructuredPatch:
        # Get horizontal coordinates
        if config.mapping == 'local':
            # LOCAL: Create a uniform grid based on mesh sizes in the dataset.
            x = np.arange(len(self.nc.dimensions['west_east'])) * self.nc.DX
            y = np.arange(len(self.nc.dimensions['south_north'])) * self.nc.DY
            x, y = np.meshgrid(x, y)
        else:
            # GLOBAL: Get the longitudes and latitudes stored in the dataset.
            x = np.deg2rad(self.variable_at('XLONG', stepid))
            y = np.deg2rad(self.variable_at('XLAT', stepid))

        nnodes = x.size
        planar_shape = tuple(s - 1 for s in x.shape)

        # Get vertical coordiantes
        if config.volumetric == 'planar':
            # PLANAR: Use the terrain height according to the dataset.
            z = self.variable_at('HGT', stepid)
        else:
            # VOLUMETRIC: Compute the height using geopotential fields.
            z = (self.variable_at('PH', stepid) + self.variable_at('PHB', stepid)) / 9.81
            x = x[__, ...]
            y = y[__, ...]
            nnodes *= self.nvertical

        # Construct the nodal array
        if config.mapping == 'local':
            # LOCAL: Straightforward insertion of x, y and z
            nodes = np.zeros(z.shape + (3,), dtype=x.dtype)
            nodes[..., 0] = x
            nodes[..., 1] = y
            nodes[..., 2] = z
            nodes = nodes.reshape((nnodes, -1))
        else:
            # GLOBAL: Add the mean Earth radius to z, then apply
            # spherical-to-Cartesian conversion.
            z += MEAN_EARTH_RADIUS
            nodes = np.array([
                z * np.cos(x) * np.cos(y),
                z * np.sin(x) * np.cos(y),
                z * np.sin(y),
            ]).reshape((-1, nnodes)).T

        # Assemble structured patch and return
        if config.volumetric == 'planar':
            return StructuredPatch(('geometry',), nodes, planar_shape, celltype=Quad())
        else:
            return StructuredPatch(('geometry',), nodes, (self.nvertical - 1,) + planar_shape, celltype=Hex())

    def write(self, w: Writer):
        # Discovert which variables to include
        if config.volumetric == 'extrude':
            allowed_types = {'volumetric', 'planar'}
        else:
            allowed_types = {config.volumetric}
        variables = [
            variable for variable in self.nc.variables
            if self.variable_type(variable) in allowed_types
        ]

        # Write data for each step
        for stepid in range(self.nsteps):
            w.add_step(time=float(stepid))

            patch = self.patch_at(stepid)
            w.update_geometry(patch)
            w.finalize_geometry()

            for variable in variables:
                data = self.variable_at(variable, stepid, config.volumetric == 'extrude')
                data = data.reshape(patch.num_nodes, -1)
                field = SimpleFieldPatch(variable, patch, data)
                w.update_field(field)

            w.finalize_step()
