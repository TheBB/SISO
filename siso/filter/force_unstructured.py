from typing import TypeVar

from ..api import Basis, Field, Step, Zone
from ..topology import DiscreteTopology, UnstructuredTopology
from .passthrough import PassthroughBFSZ


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
Z = TypeVar("Z", bound=Zone)


class ForceUnstructured(PassthroughBFSZ[B, F, S, Z, DiscreteTopology, UnstructuredTopology]):
    def validate_source(self) -> None:
        assert self.source.properties.discrete_topology

    def topology(self, timestep: S, basis: B, zone: Z) -> UnstructuredTopology:
        topology = self.source.topology(timestep, basis, zone)
        if not isinstance(topology, UnstructuredTopology):
            return UnstructuredTopology(
                num_nodes=topology.num_nodes,
                cells=topology.cells,
                celltype=topology.celltype,
                degree=topology.degree,
            )
        return topology
