from __future__ import annotations

from functools import reduce
from typing import Iterator, TypeVar, cast

from numpy import floating

from .. import util
from ..api import Basis, Field, Shape, SourceProperties, Step, Zone
from ..topology import DiscreteTopology, UnstructuredTopology
from ..util import FieldData
from .passthrough import PassthroughBFS


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
Z = TypeVar("Z", bound=Zone)


class ZoneMerge(PassthroughBFS[B, F, S, Z, Zone[int]]):
    def validate_source(self) -> None:
        assert not self.source.properties.single_zoned
        assert self.source.properties.discrete_topology

    @property
    def properties(self) -> SourceProperties:
        return super().properties.update(
            single_zoned=True,
        )

    def zones(self) -> Iterator[Zone[int]]:
        zone, multi_zone = util.first_and_has_more(self.source.zones())
        if multi_zone:
            yield Zone(
                shape=Shape.Shapeless,
                coords=(),
                key=0,
            )
        else:
            yield Zone(
                shape=zone.shape,
                coords=zone.coords,
                key=0,
            )

    def topology(self, step: S, basis: B, zone: Zone[int]) -> DiscreteTopology:
        return reduce(
            UnstructuredTopology.join,
            (cast(DiscreteTopology, self.source.topology(step, basis, z)) for z in self.source.zones()),
        )

    def field_data(self, timestep: S, field: F, zone: Zone[int]) -> FieldData[floating]:
        return FieldData.join(*(self.source.field_data(timestep, field, z) for z in self.source.zones()))
