from copy import deepcopy
from typing import Dict, Iterator, TypeVar

from .. import api, util
from ..util import FieldData
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


class Strict(Passthrough[F, T, Z, F, T, Z]):
    field_specs: Dict[str, F]
    original_properties: api.SourceProperties
    geometry: F

    def __init__(self, source: api.Source[F, T, Z]):
        super().__init__(source)
        self.field_specs = {}
        self.original_properties = deepcopy(source.properties)

    @property
    def properties(self) -> api.SourceProperties:
        properties = self.source.properties
        assert properties == self.original_properties
        return properties

    def use_geometry(self, geometry: F) -> None:
        super().use_geometry(geometry)
        self.geometry = deepcopy(geometry)

    def fields(self) -> Iterator[F]:
        for field in self.source.fields():
            if field.name not in self.field_specs:
                self.field_specs[field.name] = field
            else:
                spec = self.field_specs[field.name]
                assert spec.cellwise == field.cellwise
                assert spec.splittable == field.splittable
                assert spec.name == field.name
                assert spec.type == field.type
            yield field

    def timesteps(self) -> Iterator[T]:
        timesteps = list(self.source.timesteps())
        if self.original_properties.instantaneous:
            assert len(timesteps) == 1
        for a, b in util.pairwise(timesteps):
            assert b.index > a.index
            if b.time is not None and a.time is not None:
                assert b.time > a.time
        yield from timesteps

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        data = self.source.field_data(timestep, field, zone)
        spec = self.field_specs[field.name]
        assert spec.cellwise == field.cellwise
        assert spec.splittable == field.splittable
        assert spec.name == field.name
        assert spec.type == field.type
        assert data.ncomps == field.ncomps
        return data
