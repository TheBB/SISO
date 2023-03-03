from typing import TypeVar

from numpy import floating

from ..api import Field, SourceProperties, TimeStep
from ..topology import Topology
from ..util import FieldData
from ..zone import Zone
from .passthrough import Passthrough


Z = TypeVar("Z", bound=Zone)
F = TypeVar("F", bound=Field)
T = TypeVar("T", bound=TimeStep)


class Tesselate(Passthrough[F, T, Z, F, T, Z]):
    def validate_source(self) -> None:
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

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData[floating]:
        topology = self.source.topology(timestep, field, zone)
        tesselator = topology.tesselator()
        field_data = self.source.field_data(timestep, field, zone)
        return tesselator.tesselate_field(topology, field, field_data)
