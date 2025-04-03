from __future__ import annotations

from typing import TYPE_CHECKING

from siso import api
from siso.api import F, InB, InT, S, Z
from siso.impl import Basis

from .passthrough import PassthroughFSZ

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating

    from siso.util import FieldData

# The singleton basis object to yield.
BASIS = Basis("mesh")


class BasisMerge(PassthroughFSZ[F, S, Z, InB, Basis, InT, api.Topology]):
    """Source filter that merges bases.

    This filter will attempt (potentially unsuccessfully) to map all the fields
    from the source onto a single basis. That basis is whichever basis the
    chosen geometry is defined on.

    Parameters:
    - source: data source to draw from.
    """

    # The 'master basis' onto which everything will be mapped
    master_basis: InB

    # Keep a mapping of the merger object associated with each zone.
    mergers: dict[Z, api.TopologyMerger]

    def __init__(self, source: api.Source[InB, F, S, InT, Z]):
        super().__init__(source)
        self.mergers = {}

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(
            single_basis=True,
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

    # This method will only be called once per step and zone, because we have
    # only produced one basis object.
    def topology(self, step: S, basis: Basis, zone: Z) -> api.Topology:
        # Get the 'master topology' from the inner source.
        topology = self.source.topology(step, self.master_basis, zone)

        # Create a merger object and remember it. This assumes that the caller
        # consumes all zones before the next step.
        merger = topology.create_merger()
        self.mergers[zone] = merger

        # Call the merger to obtain the common topology we will be using.
        merged, _ = merger(topology)
        return merged

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        # To get the mapper object for transforming field data onto the new
        # topology, we first need the old topology.
        basis = self.source.basis_of(field)
        topology = self.source.topology(step, basis, zone)
        _, mapper = self.mergers[zone](topology)

        # Use the mapper to transform the field data.
        data = self.source.field_data(step, field, zone)
        return mapper(field, data)
