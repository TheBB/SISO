from __future__ import annotations

from pathlib import Path

from typing import List

import netCDF4
import numpy as np
from scipy.spatial.transform import Rotation

from .reader import Reader
from .. import config, ConfigTarget
from ..coords import Geodetic
from ..coords.util import spherical_cartesian_vf
from ..fields import Geometry, SimpleField, FieldPatches
from ..geometry import StructuredTopology, Hex, Patch
from ..util import flatten_2d


def edge_slice():
    start = config.remove_edges
    end = None if config.remove_edges == 0 else -config.remove_edges
    return slice(start, end)


class ScalarField(SimpleField):

    decompose = False
    ncomps = 1
    cells = False

    reader: SimraNcReader

    def __init__(self, name: str, reader: SimraNcReader):
        self.name = name
        self.reader = reader

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        s = edge_slice()
        patch = self.reader.patch_at(stepid)
        data = self.reader.nc[self.name][stepid, s, s, s].transpose(1, 2, 0).reshape(-1, 1)
        yield patch, data


class VectorField(SimpleField):

    components: List[str]
    decompose = False
    cells = False

    reader: SimraNcReader

    def __init__(self, name: str, components: List[str], reader: SimraNcReader):
        self.name = name
        self.components = components
        self.reader = reader
        self.ncomps = len(components)

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        s = edge_slice()

        data = np.array([
            self.reader.nc[self.components[0]][stepid, s, s, s],
            self.reader.nc[self.components[1]][stepid, s, s, s],
            self.reader.nc[self.components[2]][stepid, s, s, s],
        ]).transpose(1, 2, 3, 0)

        lon, lat = self.reader.lonlats(rotated=False)
        lon = lon[np.newaxis, ...]
        lat = lat[np.newaxis, ...]
        data = spherical_cartesian_vf(lon, lat, data)

        pole_lon = self.reader.nc['rotated_latitude_longitude'].grid_north_pole_longitude
        pole_lat = self.reader.nc['rotated_latitude_longitude'].grid_north_pole_latitude
        rotation = Rotation.from_euler('ZY', [pole_lon - 180,  pole_lat - 90], degrees=True)
        data = rotation.apply(flatten_2d(data)).reshape(data.shape)

        lon, lat = self.reader.lonlats(rotated=True)
        lon = lon[np.newaxis, ...]
        lat = lat[np.newaxis, ...]
        data = spherical_cartesian_vf(lon, lat, data, invert=True)

        yield self.reader.patch_at(stepid), flatten_2d(data)


class GeodeticGeometryField(SimpleField):

    cells = False
    ncomps = 3

    reader: SimraNcReader
    name = 'geodetic'

    def __init__(self, reader: SimraNcReader):
        self.reader = reader
        self.fieldtype = Geometry(Geodetic())

    def patches(self, stepid: int, force: bool = False, **__) -> FieldPatches:
        lon, lat = self.reader.lonlats(rotated=True)

        s = edge_slice()
        hgt = self.reader.nc['geopotential_height_ml'][stepid, s, s, s]

        nodes = np.zeros(lon.shape + (len(hgt), 3))
        nodes[..., 0] = lon[..., np.newaxis]
        nodes[..., 1] = lat[..., np.newaxis]
        nodes[..., 2] = hgt.transpose(1, 2, 0)

        yield self.reader.patch_at(stepid), flatten_2d(nodes)


class SimraNcReader(Reader):

    reader_name = "SIMRA-NC"

    nc: netCDF4.Dataset
    filename: Path

    @classmethod
    def applicable(cls, filepath: Path) -> bool:
        try:
            with netCDF4.Dataset(filepath, 'r') as f:
                assert 'simra2nc' in f.history
        except:
            return False
        return True

    def __init__(self, filename: Path):
        self.filename = filename

    def validate(self):
        super().validate()
        config.ensure_limited(ConfigTarget.Reader, 'remove_edges', reason="not supported by SIMRA-NC")

    def __enter__(self):
        self.nc = netCDF4.Dataset(self.filename, 'r').__enter__()
        return self

    def __exit__(self, *args):
        self.nc.__exit__(*args)

    def lonlats(self, rotated: bool = False):
        s = edge_slice()

        lon = self.nc['y'][s]
        lat = self.nc['x'][s]
        lon, lat = np.meshgrid(lon, lat, indexing='ij')

        if not rotated:
            return lon, lat

        pole_lon = self.nc['rotated_latitude_longitude'].grid_north_pole_longitude
        pole_lat = self.nc['rotated_latitude_longitude'].grid_north_pole_latitude

        pshape = lon.shape

        pts = np.array([
            (np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))).flatten(),
            (np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))).flatten(),
            np.sin(np.deg2rad(lat)).flatten(),
        ]).T

        rotation = Rotation.from_euler('ZY', [pole_lon - 180,  pole_lat - 90], degrees=True)
        pts = rotation.apply(pts)

        lat = np.rad2deg(np.arctan(pts[:,2] / np.sqrt(pts[:,0]**2 + pts[:,1]**2))).reshape(pshape)
        lon = np.rad2deg(np.arctan2(pts[:,1], pts[:,0])).reshape(pshape)

        return lon, lat

    def patch_at(self, stepid: int):
        s = edge_slice()
        lons = len(self.nc['y'][s])
        lats = len(self.nc['x'][s])
        hgts = len(self.nc['l'][s])
        shape = (lons - 1, lats - 1, hgts - 1)
        return Patch('geometry', StructuredTopology(shape, celltype=Hex()))

    def steps(self):
        """Iterate over all steps with associated data."""
        for stepid, time in enumerate(self.nc['time']):
            yield stepid, {'time': time}

    def variable_type(self, name: str):
        try:
            time, *dimensions = self.nc[name].dimensions
        except ValueError:
            return None

        if time != 'time':
            return None

        try:
            assert len(dimensions) == 2
            assert dimensions[0] == 'y'
            assert dimensions[1] == 'x'
            return 'planar'
        except AssertionError:
            pass

        try:
            assert len(dimensions) == 3
            assert dimensions[0] == 'l'
            assert dimensions[1] == 'y'
            assert dimensions[2] == 'x'
            return 'volumetric'
        except AssertionError:
            pass

    def fields(self):
        """Iterate over all fields."""
        yield GeodeticGeometryField(self)

        for variable in self.nc.variables:
            if self.variable_type(variable) != 'volumetric':
                continue
            yield ScalarField(variable, self)

        yield VectorField('wind', ['y_wind_ml', 'x_wind_ml', 'upward_air_velocity_ml'], self)

