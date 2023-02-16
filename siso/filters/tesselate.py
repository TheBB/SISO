from .passthrough import Passthrough
from ..api import TimeStep, Field, SourceProperties
from ..field import FieldData
from ..zone import Zone
from ..topology import Topology

from typing import TypeVar


Z = TypeVar('Z', bound=Zone)
F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)

class Tesselate(Passthrough[F, T, Z]):
    def validate_source(self):
        assert not self.source.properties.tesselated

    @property
    def properties(self) -> SourceProperties:
        return super().properties.update(
            tesselated=True,
        )

    def topology(self, timestep: T, field: F, zone: Z) -> Topology:
        topology = self.source.topology(timestep, field, zone)
        tesselator = topology.tesselator()
        return tesselator.tesselate_topology(topology)

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        topology = self.source.topology(timestep, field, zone)
        tesselator = topology.tesselator()
        field_data = self.source.field_data(timestep, field, zone)
        return tesselator.tesselate_field(topology, field, field_data)
