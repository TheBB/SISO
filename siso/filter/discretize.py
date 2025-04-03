from __future__ import annotations

from typing import TYPE_CHECKING

from siso.api import B, F, S, T, Z
from siso.topology import DiscreteTopology

from .passthrough import PassthroughBFSZ

if TYPE_CHECKING:
    from numpy import floating

    from siso import api
    from siso.util import FieldData


class Discretize(PassthroughBFSZ[B, F, S, Z, T, DiscreteTopology]):
    """Filter that discretizes all topologies, producing guaranteed either
    structured or unstructured topologies with degree 1.
    """

    nvis: int

    # When a user calls topology(), the discretization produces a mapper: a
    # callable that can be used to convert field data from old to new topology.
    # This mapper is specific to the basis and zone.
    mappers: dict[tuple[B, Z], api.FieldDataFilter]

    def __init__(self, source: api.Source[B, F, S, T, Z], nvis: int):
        super().__init__(source)
        self.nvis = nvis
        self.mappers = {}

    def validate_source(self) -> None:
        assert not self.source.properties.discrete_topology

    @property
    def properties(self) -> api.SourceProperties:
        # Pass on our guarantee of discrete topology to users.
        return self.source.properties.update(
            discrete_topology=True,
        )

    def topology(self, step: S, basis: B, zone: Z) -> DiscreteTopology:
        topology = self.source.topology(step, basis, zone)

        # Discretize the topology, and save the field data mapping for later.
        discrete, mapper = topology.discretize(self.nvis)
        self.mappers[(basis, zone)] = mapper

        return discrete

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(step, field, zone)
        basis = self.source.basis_of(field)

        # Use the stored mapper to convert field data to the new topology.
        mapper = self.mappers[(basis, zone)]
        return mapper(field, data)
