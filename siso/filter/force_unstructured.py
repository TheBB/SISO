from __future__ import annotations

from siso.api import B, F, S, Z
from siso.topology import DiscreteTopology, UnstructuredTopology

from .passthrough import PassthroughBFSZ


class ForceUnstructured(PassthroughBFSZ[B, F, S, Z, DiscreteTopology, UnstructuredTopology]):
    """Filter that converts all topologies to unstructured topologies.

    Requires that the input guarantees discrete topologies.
    """

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
