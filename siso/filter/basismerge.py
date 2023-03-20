from typing import Dict, Iterator, TypeVar

from numpy import floating

from .. import api
from ..impl import Basis
from ..util import FieldData
from .passthrough import PassthroughFSZ


InB = TypeVar("InB", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


BASIS = Basis("mesh")


class BasisMerge(PassthroughFSZ[F, S, Z, InB, Basis]):
    master_basis: InB
    mergers: Dict[Z, api.TopologyMerger]

    def __init__(self, source: api.Source[InB, F, S, Z]):
        super().__init__(source)
        self.mergers = {}

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(
            single_basis=True,
            # discrete_topology=True,
        )

    def bases(self) -> Iterator[Basis]:
        yield BASIS

    def basis_of(self, field: F) -> Basis:
        return BASIS

    def fields(self, basis: Basis) -> Iterator[F]:
        for inner_basis in self.source.bases():
            yield from self.source.fields(inner_basis)

    def geometries(self, basis: Basis) -> Iterator[F]:
        for inner_basis in self.source.bases():
            yield from self.source.geometries(inner_basis)

    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)
        self.master_basis = self.source.basis_of(geometry)

    def topology(self, step: S, basis: Basis, zone: Z) -> api.Topology:
        topology = self.source.topology(step, self.master_basis, zone)
        merger = topology.create_merger()
        self.mergers[zone] = merger
        merged, _ = merger(topology)
        return merged

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        basis = self.source.basis_of(field)
        topology = self.source.topology(step, basis, zone)
        _, mapper = self.mergers[zone](topology)
        data = self.source.field_data(step, field, zone)
        return mapper(field, data)
