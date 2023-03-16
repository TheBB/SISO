from typing import TypeVar, Generic, Iterator

from attrs import define
from numpy import floating

from .. import api
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough


Z = TypeVar("Z", bound=api.Zone)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)


BASIS = api.Basis('mesh')


@define
class WrappedField(api.Field, Generic[F]):
    original_field: F

    @property
    def name(self) -> str:
        return self.original_field.name

    @property
    def splittable(self) -> bool:
        return self.original_field.splittable

    @property
    def basis(self) -> api.Basis:
        return BASIS

    @property
    def cellwise(self) -> bool:
        return self.original_field.cellwise

    @property
    def type(self) -> api.FieldType:
        return self.original_field.type

    @property
    def ncomps(self) -> int:
        return self.original_field.ncomps


class Tesselate(Passthrough[F, S, Z, WrappedField[F], S, Z]):
    nvis: int
    master_basis: api.Basis

    def __init__(self, source: api.Source[F, S, Z], nvis: int):
        super().__init__(source)
        self.nvis = nvis
        self.master_basis = next(source.bases())

    def validate_source(self) -> None:
        assert not self.source.properties.tesselated

    @property
    def properties(self) -> api.SourceProperties:
        return super().properties.update(
            tesselated=True,
        )

    def bases(self) -> Iterator[api.Basis]:
        yield BASIS

    def geometries(self, basis: api.Basis) -> Iterator[WrappedField[F]]:
        for ibasis in self.source.bases():
            for field in self.source.geometries(ibasis):
                yield WrappedField(field)

    def fields(self, basis: api.Basis) -> Iterator[WrappedField[F]]:
        for ibasis in self.source.bases():
            for field in self.source.fields(ibasis):
                yield WrappedField(field)

    def topology(self, step: S, basis: api.Basis, zone: Z) -> Topology:
        topology = self.source.topology(step, self.master_basis, zone)
        tesselator = topology.tesselator(self.nvis)
        return tesselator.tesselate_topology(topology)

    def topology_updates(self, step: S, basis: api.Basis) -> bool:
        return self.source.topology_updates(step, self.master_basis)

    def field_data(self, step: S, field: WrappedField[F], zone: Z) -> FieldData[floating]:
        topology = self.source.topology(step, field.original_field.basis, zone)
        tesselator = topology.tesselator(self.nvis)
        field_data = self.source.field_data(step, field.original_field, zone)
        return tesselator.tesselate_field(topology, field.original_field, field_data)

    def field_updates(self, step: S, field: WrappedField[F]) -> bool:
        return super().field_updates(step, field.original_field)
