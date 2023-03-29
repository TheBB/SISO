from copy import deepcopy
from typing import Dict, Iterator, TypeVar

from numpy import floating

from .. import api, util
from ..util import FieldData
from .passthrough import PassthroughAll


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
T = TypeVar("T", bound=api.Topology)
Z = TypeVar("Z", bound=api.Zone)


class Strict(PassthroughAll[B, F, S, T, Z]):
    field_specs: Dict[str, F]
    original_properties: api.SourceProperties
    geometry: F

    def __init__(self, source: api.Source[B, F, S, T, Z]):
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

    def geometries(self, basis: B) -> Iterator[F]:
        for field in self.source.geometries(basis):
            assert field.is_geometry
            if field.name not in self.field_specs:
                self.field_specs[field.name] = field
            else:
                spec = self.field_specs[field.name]
                assert spec.cellwise == field.cellwise
                assert spec.splittable == field.splittable
                assert spec.name == field.name
                assert spec.type == field.type
            yield field

    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            assert not field.is_geometry
            if field.name not in self.field_specs:
                self.field_specs[field.name] = field
            else:
                spec = self.field_specs[field.name]
                assert spec.cellwise == field.cellwise
                assert spec.splittable == field.splittable
                assert spec.name == field.name
                assert spec.type == field.type
            yield field

    def steps(self) -> Iterator[S]:
        steps = list(self.source.steps())
        if self.original_properties.instantaneous:
            assert len(steps) == 1
        for a, b in util.pairwise(steps):
            assert b.index > a.index
            if b.value is not None and a.value is not None:
                assert b.value > a.value
        yield from steps

    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(timestep, field, zone)
        spec = self.field_specs[field.name]
        assert spec.cellwise == field.cellwise
        assert spec.splittable == field.splittable
        assert spec.name == field.name
        assert spec.type == field.type
        assert data.num_comps == field.ncomps
        return data
