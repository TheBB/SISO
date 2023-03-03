from typing import Dict, Tuple, TypeVar

from .. import api, util
from ..coords import ConversionPath, convert_coords, convert_vectors
from ..util import FieldData
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


class CoordTransform(Passthrough[F, T, Z, F, T, Z]):
    path: ConversionPath
    cache: Dict[str, FieldData]

    def __init__(self, source: api.Source[F, T, Z], path: ConversionPath):
        super().__init__(source)
        self.path = path
        self.cache = {}

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        indata = self.source.field_data(timestep, field, zone)
        if not field.is_geometry and not field.is_vector:
            return indata

        if field.is_geometry:
            for a, b in util.pairwise(self.path):
                self.cache[a.name] = indata
                indata = convert_coords(a, b, indata)
            return indata

        for a, b in util.pairwise(self.path):
            indata = convert_vectors(a, b, indata, self.cache[a.name])
        return indata
