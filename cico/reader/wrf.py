from functools import lru_cache
from pathlib import Path

import netCDF4
from numpy import newaxis as __
import numpy as np
from scipy.spatial.transform import Rotation
import treelog as log

from typing import Optional, Tuple, Iterable, List
from ..typing import Shape, Array2D, StepData

from .. import config, ConfigTarget
from .reader import Reader
from ..writer import Writer
from ..geometry import Quad, Hex, Patch, StructuredPatch, UnstructuredPatch
from ..fields import Field, FieldPatch, SimpleFieldPatch
from ..util import unstagger, structured_cells, angle_mean_deg, nodemap as mknodemap


MEAN_EARTH_RADIUS = 6_371_000


# WRF data sets contain many forms of data, which makes this reader
# complicated.  The main options are as follows:
#
# config.volumetric:
# - if 'volumetric', the output will be a volumetric mesh with all 3D fields
# - if 'planar', the output will be a surface mesh with all 2D fields,
#   including the surface slice of all 3D fields
# - if 'extrude', the output will be a volumetric mesh with all 3D fields,
#   including 2D fields that will simply be constant in the vertical direction
#
# config.mapping:
# - if 'local', the output will be in physical projected coordinates,
#   derived from the DX and DY attributes in the data file, suitable
#   if the computational domain is small with respect to the size of
#   the Earth
# - if 'global', we will attempt to convert from latitude/longitude
#   coordinates to 'true' Cartesian coordiantes, where the horizon is
#   in the XY plane, the Z axis points toward the north pole and the X
#   axis points toward the intersection between the prime meridian and
#   the equator
#
# config.periodic:
# - if true, we will tie the mesh together at the 'missing' meridian
#   and the poles, so that the output looks like a closed sphere
#   rather than a sphere with some cut-out areas
#
# Except for config.periodic implying config.mapping == 'global',
# these options are all independent, creating a lovely mess.  I have
# tried to clarify as much as possible what is happening and why.
#
# For the global mapping, we use the XLONG and XLAT variables in the
# data file for placing the mesh nodes.  Vector fields (such as wind)
# must also be rotated.  This happens in two steps:
#
# 1. Vectors are converted from the raw coordinates in the data file
#    to Cartesian coordinates.
# 2. Because the grid poles and meridian may not match the true poles
#    and meridians, the resulting vectors are then rotated to the final
#    coordinate system.
#
# Good luck.


class WRFScalarField(Field):

    reader: 'WRFReader'
    decompose = False
    ncomps = 1
    cells = False

    def __init__(self, name: str, reader: 'WRFReader'):
        self.name = name
        self.reader = reader

    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        patch = self.reader.patch_at(stepid)
        data = self.reader.variable_at(self.name, stepid, config.volumetric == 'extrude')
        data = data.reshape(patch.num_nodes, -1)
        yield SimpleFieldPatch(self.name, patch, data)


class WRFVectorField(Field):

    reader: 'WRFReader'
    components: List[str]
    decompose = False
    cells = False

    def __init__(self, name: str, components: List[str], reader: 'WRFReader'):
        self.name = name
        self.components = components
        self.reader = reader
        self.ncomps = len(components)

    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        patch = self.reader.patch_at(stepid)
        yield self.reader.velocity_field(patch, stepid)



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

    def validate(self):
        super().validate()

        # Disable periodicity except in global mapping
        if config.mapping == 'local':
            config.require(periodic=False, reason="WRF does not support periodic local grids, try with --global")
        else:
            log.warning("Global mapping of WRF data is experimental, please do not use indiscriminately")

        config.ensure_limited(
            ConfigTarget.Reader, 'volumetric', 'mapping', 'periodic',
            reason="not supported by WRF"
        )

    def __enter__(self):
        self.nc = netCDF4.Dataset(self.filename, 'r').__enter__()
        return self

    def __exit__(self, *args):
        self.nc.__exit__(*args)

    @property
    def nsteps(self) -> int:
        """Number of time steps in the dataset."""
        return len(self.nc.dimensions['Time'])

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        for stepid in range(self.nsteps):
            yield stepid, {'time': self.nc['XTIME'][stepid] * 60}

    @property
    def nlat(self) -> int:
        """Number of points in latitudinal direction."""
        return len(self.nc.dimensions['south_north'])

    @property
    def nlon(self) -> int:
        """Number of points in longitudinal direction."""
        return len(self.nc.dimensions['west_east'])

    @property
    def nplanar(self) -> int:
        """Number of points in a single grid plane, not including possible
        polar points that are appended later.
        """
        return self.nlat * self.nlon

    @property
    def planar_shape(self) -> Shape:
        """Shape (in terms of cells) of a single grid plane."""
        return (self.nlat - 1, self.nlon - 1)

    @property
    def nvert(self) -> int:
        """Number of points in vertical direction."""
        return len(self.nc.dimensions['bottom_top'])

    @property
    def volumetric_shape(self) -> Shape:
        """Shape (in terms of cells) of the volumetric grid."""
        return (self.nvert - 1, *self.planar_shape)

    def rotation(self) -> Rotation:
        """Return a rotation that sends the north pole to the grid's
        north pole, the south pole to the grid's south pole and the
        prime meridian to the grid's prime meridian.

        This is step 2 of the two-step coordinate transform outlined
        in the beginning of the file.
        """

        # Note: the final rotation around Z is essentially guesswork.
        intrinsic = 360 * np.ceil(self.nlon / 2) / self.nlon
        return Rotation.from_euler('ZYZ', [-self.nc.STAND_LON, -self.nc.MOAD_CEN_LAT, intrinsic], degrees=True)

    def cartesian_field(self, data: Array2D) -> Array2D:
        """Convert a vector field in spherical coordinates to a vector
        field in Cartesian coordinates.  This is achieved by
        multiplying with a pointwise 3D rotation matrix computed from
        assumption of the grid's spherical coordinates.  This probably
        only works for Cylindrical Equidistant projection.

        The data must be structured, i.e. the polar points should not
        be included.
        """

        # Convert to structured shape
        data = data.reshape((-1, self.nlat, self.nlon, 3))

        # Compute the grid's spherical coordinates and assemble a
        # pointwise rotation matrix
        lon = np.deg2rad(np.linspace(0, 360, self.nlon, endpoint=False)[__, :])
        lat = np.deg2rad(np.linspace(-90, 90, 2 * self.nlat + 1)[1::2][:, __])
        clon, clat = np.cos(lon), np.cos(lat)
        slon, slat = np.sin(lon), np.sin(lat)
        lon1, lat1 = np.ones_like(lon), np.ones_like(lat)

        rot = np.array([
            [-slon * lat1, -slat * clon, clat * clon],
            [clon * lat1, -slat * slon, clat * slon],
            [np.zeros_like(lat * lon), clat * lon1, slat * lon1]
        ])

        # Apply rotation to data
        return np.einsum('mnjk,ijkn->ijkm', rot, data)

    def variable_at(self, name: str, stepid: int,
                    extrude_if_planar: bool = False,
                    include_poles: bool = True) -> np.ndarray:
        """Extract a variable with a given name at a given time from
        the dataset.

        If 'extrude_if_planar' is set, planar variables will be
        extruded with constant values in the vertical direction.

        If 'include_poles' is set, calculate the values at the polar
        points by averaging the values nearby, and include them.

        The returned array has the vertical axis on the first
        dimension, and the horizontal axes flattened on the second
        dimension.  All variables in the dataset should be scalar, so
        there is no third dimension.
        """

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

        # Compute polar values if necessary
        if include_poles and config.periodic:
            if name in ('XLONG', 'XLAT'):
                south = angle_mean_deg(data[0])
                north = angle_mean_deg(data[-1])
            else:
                south = np.mean(data[...,  0, :], axis=-1)
                north = np.mean(data[..., -1, :], axis=-1)

        # Flatten the horizontals but leave the verticals intact
        if len(dimensions) == 3:
            data = data.reshape((self.nvert, -1))
        else:
            data = data.flatten()

        # If periodic, append previously computed polar values
        if include_poles and config.periodic:
            appendix = np.array([south, north]).T
            data = np.append(data, appendix, axis=-1)

        # Extrude the vertical direction if desired
        if extrude_if_planar and len(dimensions) == 2:
            newdata = np.zeros((self.nvert,) + data.shape, dtype=data.dtype)
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

    @lru_cache(1)
    def patch_at(self, stepid: int) -> Patch:
        """Construct the patch object at the given time step.  This method
        handles all variations of mesh options.
        """

        # Get horizontal coordinates
        if config.mapping == 'local':
            # LOCAL: Create a uniform grid based on mesh sizes in the dataset.
            x = np.arange(self.nlon) * self.nc.DX
            y = np.arange(self.nlat) * self.nc.DY
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
            nnodes *= self.nvert

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
        # are stored in order: S/N, W/E
        cells = structured_cells(self.planar_shape, 2)

        # Append a layer of cells for periodicity in the longitude direction
        nodemap = mknodemap((self.nlat, 2), (self.nlon, self.nlon - 1))
        appendix = structured_cells((self.nlat - 1, 1), 2, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the southern boundary to the south pole
        pole_id = self.nplanar
        nodemap = mknodemap((2, self.nlon + 1), (pole_id, 1), periodic=(1,))
        nodemap[1] = nodemap[1,0]
        appendix = structured_cells((1, self.nlon), 2, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the northern boundary to the north pole
        pole_id = self.nplanar + 1
        nodemap = mknodemap((2, self.nlon + 1), (-self.nlon - 1, 1), periodic=(1,), init=pole_id)
        nodemap[0] = nodemap[0,0]
        appendix = structured_cells((1, self.nlon), 2, nodemap)
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
        cells += cells // (self.nlat * self.nlon) * 2

        # Append a layer of cells for periodicity in the longitude direction
        nhoriz = self.nplanar + 2
        nodemap = mknodemap((self.nvert, self.nlat, 2), (nhoriz, self.nlon, self.nlon - 1))
        appendix = structured_cells((self.nvert - 1, self.nlat - 1, 1), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the southern boundary to the south pole
        pole_id = self.nplanar
        nodemap = mknodemap((self.nvert, 2, self.nlon + 1), (self.nplanar + 2, pole_id, 1), periodic=(2,))
        nodemap[:,1] = (nodemap[:,1] - pole_id) // nhoriz * nhoriz + pole_id
        appendix = structured_cells((self.nvert - 1, 1, self.nlon), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        # Append a layer of cells tying the northern boundary to the north pole
        pole_id = self.nplanar + 1
        nodemap = mknodemap((self.nvert, 2, self.nlon + 1), (nhoriz, -self.nlon - 1, 1), periodic=(2,), init=pole_id)
        nodemap[:,0] = (nodemap[:,0] - pole_id) // nhoriz * nhoriz + pole_id
        appendix = structured_cells((self.nvert - 1, 1, self.nlon), 3, nodemap)
        cells = np.append(cells, appendix, axis=0)

        return cells

    def velocity_field(self, patch: Patch, stepid: int) -> SimpleFieldPatch:
        """Compute the velocity field at a given time step.

        In the simplest case, this is just a matter of concatenating
        the U, V and W data sets.  For global mapping, we must also
        transform the vector field accordingly.
        """

        # Extract raw data.  For global mapping, compute the polar
        # values AFTER transformation.
        kwargs = {
            'include_poles': config.mapping == 'local',
            'extrude_if_planar': config.volumetric == 'extrude',
        }
        data = np.array([self.variable_at(x, stepid, **kwargs).reshape(-1) for x in 'UVW']).T

        # For local mapping, we're done
        if config.mapping == 'local':
            return SimpleFieldPatch('WIND', patch, data)

        # Convert spherical coordinates to the grid's own Cartesian coordinate system
        data = self.cartesian_field(data)

        # Extract mean values at poles
        if config.periodic:
            south = np.mean(data[:, 0, ...], axis=-2)[:, __, :]
            north = np.mean(data[:, -1, ...], axis=-2)[:, __, :]

        # Flatten the horizontal directions
        data = data.reshape((-1, self.nlat * self.nlon, 3))

        # Append mean values at poles
        if config.periodic:
            data = np.append(data, south, axis=1)
            data = np.append(data, north, axis=1)

        # Rotate to final coordinate system
        data = data.reshape(-1, 3)
        data = self.rotation().apply(data)
        return SimpleFieldPatch('WIND', patch, data)

    def geometry(self, stepid: int, force: bool = False) -> Iterable[Patch]:
        yield self.patch_at(stepid)

    def fields(self) -> Iterable[Field]:
        if config.volumetric == 'volumetric':
            allowed_types = {'volumetric'}
        else:
            allowed_types = {'volumetric', 'planar'}

        for variable in self.nc.variables:
            if self.variable_type(variable) in allowed_types:
                yield WRFScalarField(variable, self)

        if all(self.variable_type(x) in allowed_types for x in 'UVW'):
            yield WRFVectorField('WIND', ['U', 'V', 'W'], self)
