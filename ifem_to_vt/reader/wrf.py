from pathlib import Path

import netCDF4
from numpy import newaxis as __
import numpy as np

from typing import Optional
from ..typing import Shape

from .. import config
from .reader import Reader
from ..writer import Writer
from ..geometry import Quad, Hex, StructuredPatch, UnstructuredPatch
from ..fields import SimpleFieldPatch
from ..util import unstagger, structured_cells, angle_mean_deg, nodemap as mknodemap


MEAN_EARTH_RADIUS = 6_371_000


class WRFReader(Reader):

    reader_name = "WRF"

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
    def nlatitude(self) -> int:
        return len(self.nc.dimensions['south_north'])

    @property
    def nlongitude(self) -> int:
        return len(self.nc.dimensions['west_east'])

    @property
    def nplanar(self) -> int:
        return self.nlatitude * self.nlongitude

    @property
    def nvertical(self) -> int:
        return len(self.nc.dimensions['bottom_top'])

    @property
    def planar_shape(self) -> Shape:
        return (self.nlatitude - 1, self.nlongitude - 1)

    @property
    def volumetric_shape(self) -> Shape:
        return (self.nvertical - 1, *self.planar_shape)

    def variable_at(self, name: str, stepid: int, extrude_if_planar: bool = False) -> np.ndarray:
        time, *dimensions = self.nc[name].dimensions
        dimensions = list(dimensions)
        assert time == 'Time'
        data = self.nc[name][stepid, ...]

        # If we're in planar mode and the field is volumetric, grab
        # the surface slice
        if len(dimensions) == 3 and config.volumetric == 'planar':
            data = data[0, ...]
            dimensions = dimensions[1:]

        # Detect staggered axes and un-stagger them
        for i, dim in enumerate(dimensions):
            if dim.endswith('_stag'):
                data = unstagger(data, i)
                dimensions[i] = dim[:-5]

        # If periodic, compute polar values
        if config.periodic and name in ('XLONG', 'XLAT'):
            south = angle_mean_deg(data[0])
            north = angle_mean_deg(data[-1])
        elif config.periodic:
            south = np.mean(data[...,  0, :], axis=-1)
            north = np.mean(data[..., -1, :], axis=-1)

        # Flatten the horizontals but leave the verticals intact
        if len(dimensions) == 3:
            data = data.reshape((self.nvertical, -1))
        else:
            data = data.flatten()

        # If periodic, append previously computed polar values
        if config.periodic:
            appendix = np.array([south, north]).T
            data = np.append(data, appendix, axis=-1)

        # Extrude the vertical direction if desired
        if extrude_if_planar and len(dimensions) == 2:
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
            x = np.arange(self.nlongitude) * self.nc.DX
            y = np.arange(self.nlatitude) * self.nc.DY
            x, y = np.meshgrid(x, y)
            x = x.flatten()
            y = y.flatten()
        else:
            # GLOBAL: Get the longitudes and latitudes stored in the dataset.
            x = np.deg2rad(self.variable_at('XLONG', stepid))
            y = np.deg2rad(self.variable_at('XLAT', stepid))

        nnodes = x.size

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

        # Assemble structured or unstructured patch and return
        if config.periodic and config.volumetric == 'planar':
            return UnstructuredPatch(('geometry',), nodes, self.periodic_planar_mesh(), celltype=Quad())
        elif config.periodic:
            return UnstructuredPatch(('geometry',), nodes, self.periodic_volumetric_mesh(), celltype=Hex())
        elif config.volumetric == 'planar':
            return StructuredPatch(('geometry',), nodes, self.planar_shape, celltype=Quad())
        else:
            return StructuredPatch(('geometry',), nodes, self.volumetric_shape, celltype=Hex())

    def periodic_planar_mesh(self):
        """Compute cell topology for the periodic planar unstructured case,
        with polar points.
        """
        assert config.periodic
        assert config.volumetric == 'planar'

        # Construct the basic structured mesh.  Note that our nodes
        # are stored in order: vertical, S/N, W/E
        cells = structured_cells(self.planar_shape, 2)

        # Append a layer of cells for periodicity in the longitude direction
        nodemap = mknodemap((self.nlatitude, 2), (self.nlongitude, self.nlongitude - 1))
        appendix = structured_cells((self.nlatitude - 1, 1), 2, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the southern boundary to the south pole
        pole_id = self.nplanar
        nodemap = mknodemap((2, self.nlongitude + 1), (pole_id, 1), periodic=(1,))
        nodemap[1] = nodemap[1,0]
        appendix = structured_cells((1, self.nlongitude), 2, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the northern boundary to the north pole
        pole_id = self.nplanar + 1
        nodemap = mknodemap((2, self.nlongitude + 1), (-self.nlongitude - 1, 1), periodic=(1,), init=pole_id)
        nodemap[0] = nodemap[0,0]
        appendix = structured_cells((1, self.nlongitude), 2, nodemap)
        cells = np.append(cells, appendix, axis=0)

        return cells

    def periodic_volumetric_mesh(self):
        """Compute cell topology for the periodic volumetric unstructured
        case, with polar points.
        """
        assert config.periodic
        assert config.volumetric != 'planar'

        # Construct the basic structured mesh.  Note that our nodes
        # are stored in order: vertical, S/N, W/E
        cells = structured_cells(self.volumetric_shape, 3)

        # Increment indices by two for every vertical layer, to
        # account for the polar points
        cells += cells // (self.nlatitude * self.nlongitude) * 2

        # Append a layer of cells for periodicity in the longitude direction
        nodemap = mknodemap((self.nvertical, self.nlatitude, 2), (173, 19, 18))
        appendix = structured_cells((self.nvertical - 1, self.nlatitude - 1, 1), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the southern boundary to the south pole
        nodemap = mknodemap((self.nvertical, 2, self.nlongitude + 1), (173, 171, 1), periodic=(2,))
        nodemap[:,1] = (nodemap[:,1] - 171) // 173 * 173 + 171
        appendix = structured_cells((self.nvertical - 1, 1, self.nlongitude), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the northern boundary to the north pole
        nodemap = mknodemap((self.nvertical, 2, self.nlongitude + 1), (173, -20, 1), periodic=(2,), init=172)
        nodemap[:,0] = (nodemap[:,0] - 172) // 173 * 173 + 172
        appendix = structured_cells((self.nvertical - 1, 1, self.nlongitude), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        return cells

    def write(self, w: Writer):
        # Discovert which variables to include
        if config.volumetric == 'volumetric':
            allowed_types = {'volumetric'}
        else:
            allowed_types = {'volumetric', 'planar'}
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
