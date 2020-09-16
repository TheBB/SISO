from abc import abstractmethod
from functools import lru_cache
from pathlib import Path

import netCDF4
from numpy import newaxis as __
import numpy as np
from scipy.spatial.transform import Rotation
import treelog as log

from typing import Optional, Tuple, Iterable, List
from ..typing import Shape, Array2D, StepData

from .reader import Reader
from .. import config, ConfigTarget
from ..coords import Local, Geocentric, Geodetic, Coords
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import Quad, Hex, Patch, StructuredPatch, UnstructuredPatch
from ..util import unstagger, structured_cells, angle_mean_deg, nodemap as mknodemap, flatten_2d, spherical_cartesian_vf
from ..writer import Writer



class WRFScalarField(SimpleField):

    decompose = False
    ncomps = 1
    cells = False

    reader: 'WRFReader'

    def __init__(self, name: str, reader: 'WRFReader'):
        self.name = name
        self.reader = reader

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        patch = self.reader.patch_at(stepid)
        kwargs = {'extrude_if_planar': config.volumetric == 'extrude'}
        data = self.reader.variable_at(self.name, stepid, **kwargs)
        yield patch, data.reshape(patch.num_nodes, -1)


class WRFVectorField(SimpleField):

    components: List[str]
    decompose = False
    cells = False

    reader: 'WRFReader'

    def __init__(self, name: str, components: List[str], reader: 'WRFReader'):
        self.name = name
        self.components = components
        self.reader = reader
        self.ncomps = len(components)

    def patches(self, stepid: int, force: bool = False, coords: Optional[Coords] = None) -> FieldPatches:
        kwargs = {'extrude_if_planar': config.volumetric == 'extrude'}

        if isinstance(coords, Local):
            data = np.array([self.reader.variable_at(x, stepid, **kwargs).flatten() for x in 'UVW']).T
            yield self.reader.patch_at(stepid), data; return

        data = np.array([self.reader.variable_at(x, stepid, include_poles=False, **kwargs).flatten() for x in 'UVW']).T
        reader = self.reader

        # Convert to structured shape
        data = data.reshape((-1, reader.nlat, reader.nlon, 3))

        # Convert to rotated geocentric coordinates
        lon = np.linspace(0, 360, reader.nlon, endpoint=False)[__, :]
        lat = np.linspace(-90, 90, 2 * reader.nlat + 1)[1::2][:, __]
        data = spherical_cartesian_vf(lon, lat, data)

        # Extract mean values at poles
        if config.periodic:
            south = np.mean(data[:, 0, ...], axis=-2)[:, __, :]
            north = np.mean(data[:, -1, ...], axis=-2)[:, __, :]

        # Flatten the horizontal directions
        data = data.reshape((-1, reader.nlat * reader.nlon, 3))

        # Append mean values at poles
        if config.periodic:
            data = np.append(data, south, axis=1)
            data = np.append(data, north, axis=1)

        # Rotate to true geocentric coordinates
        data = reader.rotation().apply(flatten_2d(data)).reshape(data.shape)

        # Convert back to geodetic coordinates
        lon = self.reader.variable_at('XLONG', stepid)
        lat = self.reader.variable_at('XLAT', stepid)
        data = spherical_cartesian_vf(lon, lat, data, invert=True)

        yield self.reader.patch_at(stepid), flatten_2d(data)


class WRFGeometryField(SimpleField):

    cells = False
    ncomps = 3

    reader: 'WRFReader'

    def __init__(self, reader: 'WRFReader'):
        self.reader = reader

    @abstractmethod
    def nodes(self, stepid: int) -> Array2D:
        pass

    def height(self, stepid: int, x: Array2D, y: Array2D) -> Tuple[Array2D, Array2D, Array2D]:
        if config.volumetric == 'planar':
            # PLANAR: Use the terrain height according to the dataset.
            return x, y, self.reader.variable_at('HGT', stepid)
        # VOLUMETRIC: Compute the height using geopotential fields.
        z = (self.reader.variable_at('PH', stepid) + self.reader.variable_at('PHB', stepid)) / 9.81
        return x[__, ...], y[__, ...], z

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        x, y, z = self.nodes(stepid)
        nodes = np.zeros(z.shape + (3,), dtype=x.dtype)
        nodes[..., 0] = x
        nodes[..., 1] = y
        nodes[..., 2] = z
        yield self.reader.patch_at(stepid), flatten_2d(nodes)


class WRFLocalGeometryField(WRFGeometryField):

    cells = False
    ncomps = 3

    def __init__(self, reader: 'WRFReader'):
        super().__init__(reader)
        self.fieldtype = Geometry(Local())
        self.name = 'local'

    def nodes(self, stepid: int) -> Array2D:
        reader = self.reader
        x = np.arange(reader.nlon) * reader.nc.DX
        y = np.arange(reader.nlat) * reader.nc.DY
        x, y = np.meshgrid(x, y)
        return self.height(stepid, x.flatten(), y.flatten())


class WRFGeodeticGeometryField(WRFGeometryField):

    def __init__(self, reader: 'WRFReader'):
        super().__init__(reader)
        self.fieldtype = Geometry(Geodetic())
        self.name = 'geodetic'

    def nodes(self, stepid: int) -> Array2D:
        lon = self.reader.variable_at('XLONG', stepid)
        lat = self.reader.variable_at('XLAT', stepid)
        return self.height(stepid, lon, lat)


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

        # Disable periodicity except in geocentric coordinates
        if not isinstance(config.coords, Geocentric):
            config.require(
                periodic=False,
                reason="WRF does not support periodic non-geocentric coordinates; try with --coords geocentric"
            )
        else:
            log.warning("Geocentric coordinates of WRF data is experimental, please do not use indiscriminately")

        config.ensure_limited(
            ConfigTarget.Reader, 'volumetric', 'periodic',
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

    def variable_at(self, name: str, stepid: int,
                    include_poles: bool = True,
                    extrude_if_planar: bool = False) -> np.ndarray:
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

        # Detect staggered axes and un-stagger them
        for i, dim in enumerate(dimensions):
            if dim.endswith('_stag'):
                data = unstagger(data, i)
                dimensions[i] = dim[:-5]

        # If we're in planar mode and the field is volumetric, grab
        # the surface slice
        if len(dimensions) == 3 and config.volumetric == 'planar':
            index = len(self.nc.dimensions['soil_layers_stag']) - 1
            data = data[index, ...]
            dimensions = dimensions[1:]

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

    def patch_at(self, stepid: int) -> Patch:
        """Construct the patch object at the given time step.  This method
        handles all variations of mesh options.
        """

        nnodes = self.nplanar
        if config.periodic:
            nnodes += 2
        if config.volumetric != 'planar':
            nnodes *= self.nvert

        if config.periodic and config.volumetric == 'planar':
            return UnstructuredPatch(('geometry',), nnodes, self.periodic_planar_mesh(), celltype=Quad())
        elif config.periodic:
            return UnstructuredPatch(('geometry',), nnodes, self.periodic_volumetric_mesh(), celltype=Hex())
        elif config.volumetric == 'planar':
            return StructuredPatch(('geometry',), self.planar_shape, celltype=Quad())
        else:
            return StructuredPatch(('geometry',), self.volumetric_shape, celltype=Hex())

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

    def fields(self) -> Iterable[Field]:
        yield WRFLocalGeometryField(self)
        yield WRFGeodeticGeometryField(self)

        if config.volumetric == 'volumetric':
            allowed_types = {'volumetric'}
        else:
            allowed_types = {'volumetric', 'planar'}

        for variable in self.nc.variables:
            if self.variable_type(variable) in allowed_types:
                yield WRFScalarField(variable, self)

        if all(self.variable_type(x) in allowed_types for x in 'UVW'):
            yield WRFVectorField('WIND', ['U', 'V', 'W'], self)
