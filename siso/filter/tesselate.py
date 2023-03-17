from typing import Iterator, TypeVar

from attrs import define
from numpy import floating

from .. import api
from ..impl import Basis
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough, WrappedField


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


BASIS = Basis("mesh")


@define
class Wrapped(WrappedField[F]):
    original_field: F


class Tesselate(Passthrough[B, F, S, Z, Basis, Wrapped[F], S, Z]):
    nvis: int
    master_basis: B

    def __init__(self, source: api.Source[B, F, S, Z], nvis: int):
        super().__init__(source)
        self.nvis = nvis

    def validate_source(self) -> None:
        assert not self.source.properties.tesselated

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(
            tesselated=True,
        )

    def bases(self) -> Iterator[Basis]:
        yield BASIS

    def basis_of(self, field: Wrapped[F]) -> Basis:
        return BASIS

    def use_geometry(self, geometry: Wrapped[F]) -> None:
        self.source.use_geometry(geometry.original_field)
        self.master_basis = self.source.basis_of(geometry.original_field)

    def geometries(self, basis: Basis) -> Iterator[Wrapped[F]]:
        for ibasis in self.source.bases():
            for field in self.source.geometries(ibasis):
                yield Wrapped(field)

    def fields(self, basis: Basis) -> Iterator[Wrapped[F]]:
        for ibasis in self.source.bases():
            for field in self.source.fields(ibasis):
                yield Wrapped(field)

    def topology(self, step: S, basis: Basis, zone: Z) -> Topology:
        topology = self.source.topology(step, self.master_basis, zone)
        tesselator = topology.tesselator(self.nvis)
        return tesselator.tesselate_topology(topology)

    def topology_updates(self, step: S, basis: Basis) -> bool:
        return self.source.topology_updates(step, self.master_basis)

    def field_data(self, step: S, field: Wrapped[F], zone: Z) -> FieldData[floating]:
        basis = self.source.basis_of(field.original_field)
        topology = self.source.topology(step, basis, zone)
        tesselator = topology.tesselator(self.nvis)
        field_data = self.source.field_data(step, field.original_field, zone)
        return tesselator.tesselate_field(topology, field.original_field, field_data)

    def field_updates(self, step: S, field: Wrapped[F]) -> bool:
        return self.source.field_updates(step, field.original_field)
