from typing import TypeVar, Iterator

from attrs import define
from numpy import floating

from .. import api
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough, WrappedField


Z = TypeVar("Z", bound=api.Zone)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)


BASIS = api.Basis('mesh')


@define
class Wrapped(WrappedField[F]):
    original_field: F

    @property
    def basis(self) -> api.Basis:
        return BASIS


class Tesselate(Passthrough[F, S, Z, Wrapped[F], S, Z]):
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

    def geometries(self, basis: api.Basis) -> Iterator[Wrapped[F]]:
        for ibasis in self.source.bases():
            for field in self.source.geometries(ibasis):
                yield Wrapped(field)

    def fields(self, basis: api.Basis) -> Iterator[Wrapped[F]]:
        for ibasis in self.source.bases():
            for field in self.source.fields(ibasis):
                yield Wrapped(field)

    def topology(self, step: S, basis: api.Basis, zone: Z) -> Topology:
        topology = self.source.topology(step, self.master_basis, zone)
        tesselator = topology.tesselator(self.nvis)
        return tesselator.tesselate_topology(topology)

    def topology_updates(self, step: S, basis: api.Basis) -> bool:
        return self.source.topology_updates(step, self.master_basis)

    def field_data(self, step: S, field: Wrapped[F], zone: Z) -> FieldData[floating]:
        topology = self.source.topology(step, field.original_field.basis, zone)
        tesselator = topology.tesselator(self.nvis)
        field_data = self.source.field_data(step, field.original_field, zone)
        return tesselator.tesselate_field(topology, field.original_field, field_data)

    def field_updates(self, step: S, field: Wrapped[F]) -> bool:
        return self.source.field_updates(step, field.original_field)
