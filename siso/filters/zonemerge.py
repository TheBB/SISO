from __future__ import annotations

from functools import reduce

from .passthrough import Passthrough
from ..api import Field, TimeStep, SourceProperties
from ..field import FieldData
from ..topology import DiscreteTopology, UnstructuredTopology
from ..zone import Zone, Shape
from .. import util

from typing import (
    cast,
    Iterator,
    TypeVar,
)


F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)

class ZoneMerge(Passthrough[F, T, Zone]):
    def validate_source(self) -> None:
        assert not self.source.properties.single_zoned
        assert self.source.properties.tesselated

    @property
    def properties(self) -> SourceProperties:
        return super().properties.update(
            single_zoned=True,
        )

    def zones(self) -> Iterator[Zone]:
        zone, multi_zone = util.first_and_has_more(self.source.zones())
        if multi_zone:
            yield Zone(
                shape=Shape.Shapeless,
                coords=(),
                local_key='superzone',
                global_key=0,
            )
        else:
            yield Zone(
                shape=zone.shape,
                coords=zone.coords,
                local_key='superzone',
                global_key=0,
            )

    def topology(self, timestep: T, field: F, zone: Zone) -> DiscreteTopology:
        return reduce(
            UnstructuredTopology.join,
            (
                cast(DiscreteTopology, self.source.topology(timestep, field, z))
                for z in self.source.zones()
            ),
        )

    def field_data(self, timestep: T, field: F, zone: Zone) -> FieldData:
        return reduce(
            lambda x, y: x.join(y),
            (self.source.field_data(timestep, field, z) for z in self.source.zones()),
        )
