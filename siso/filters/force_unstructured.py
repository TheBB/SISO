from .passthrough import Passthrough
from ..api import TimeStep, Field
from ..topology import DiscreteTopology, UnstructuredTopology
from ..zone import Zone

from typing import cast, TypeVar


Z = TypeVar('Z', bound=Zone)
F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)

class ForceUnstructured(Passthrough[F, T, Z]):
    def validate_source(self):
        assert self.source.properties.tesselated

    def topology(self, timestep: T, field: F, zone: Z) -> UnstructuredTopology:
        topology = cast(DiscreteTopology, self.source.topology(timestep, field, zone))
        if not isinstance(topology, UnstructuredTopology):
            return UnstructuredTopology(
                num_nodes=topology.num_nodes,
                cells=topology.cells,
                celltype=topology.celltype,
            )
        return topology
