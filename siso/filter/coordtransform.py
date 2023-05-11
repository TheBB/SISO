from typing import Dict, Generic, Tuple

from numpy import floating

from .. import api, util
from ..api import B, F, S, T, Z
from ..coord import ConversionPath, convert_coords, convert_vectors
from ..util import FieldData
from .passthrough import PassthroughAll


class CoordTransformCache(Generic[Z]):
    path: ConversionPath

    # Conversion of vector fields requires (in general) knowledge of the
    # coordinates of the points those vector fields are defined on. We store
    # them in this dict, mapping coordinate system name and zone to nodes.
    cache: Dict[Tuple[str, Z], FieldData[floating]]

    def __init__(self, path: ConversionPath) -> None:
        self.path = path
        self.cache = {}

    def convert_geometry(self, data: FieldData[floating], zone: Z) -> FieldData[floating]:
        for a, b in util.pairwise(self.path):
            self.cache[a.name, zone] = data
            data = convert_coords(a, b, data)
        return data

    def convert_vectors(self, data: FieldData[floating], zone: Z) -> FieldData[floating]:
        for a, b in util.pairwise(self.path):
            data = convert_vectors(a, b, data, self.cache[a.name, zone])
        return data


class CoordTransform(PassthroughAll[B, F, S, T, Z]):
    """Coordinate transform filter.

    This filter converts geometries and vector fields from one coordinate system
    (defined by the source geometry), to another.

    Parameters:
    - source: data source to draw from.
    - path: the coordinate system conversion path to use. Generated either by
        `siso.coord.conversion_path` or `siso.coord.optimal_system`.
    """

    cache: CoordTransformCache[Z]

    def __init__(self, source: api.Source[B, F, S, T, Z], path: ConversionPath):
        super().__init__(source)
        self.cache = CoordTransformCache(path)

    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(timestep, field, zone)

        # Coordinate conversion leaves scalar fields alone.
        if not field.is_geometry and not field.is_vector:
            return data

        # For geometries, convert and store in cache.
        if field.is_geometry:
            return self.cache.convert_geometry(data, zone)

        # For vectors, use cached data to convert.
        return self.cache.convert_vectors(data, zone)
