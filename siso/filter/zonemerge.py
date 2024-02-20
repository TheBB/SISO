from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from siso import util
from siso.api import B, F, Points, S, SourceProperties, Z, Zone, ZoneShape
from siso.topology import DiscreteTopology, UnstructuredTopology
from siso.util import FieldData

from .passthrough import PassthroughBFST

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating


class ZoneMerge(PassthroughBFST[B, F, S, DiscreteTopology, Z, Zone[int]]):
    """Filter that merges all zones into one.

    This requires that the source data is discrete.
    """

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
            # We can't make any predictions about the shape of a merged zone.
            yield Zone(
                shape=ZoneShape.Shapeless,
                coords=Points(()),
                key=0,
            )
        else:
            # ...but if the source data only has a single zone, reproduce it as
            # much as we can.
            yield Zone(
                shape=zone.shape,
                coords=zone.coords,
                key=0,
            )

    def topology(self, step: S, basis: B, zone: Zone[int]) -> DiscreteTopology:
        # Join all topologies into an unstructured topology. This should produce
        # a structured topology if the source data only has one (and it is
        # structured).
        return reduce(
            UnstructuredTopology.join,
            (self.source.topology(step, basis, z) for z in self.source.zones()),
        )

    def field_data(self, timestep: S, field: F, zone: Zone[int]) -> FieldData[floating]:
        return FieldData.join_dofs(*(self.source.field_data(timestep, field, z) for z in self.source.zones()))
