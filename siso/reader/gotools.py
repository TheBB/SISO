from __future__ import annotations

from ..topology import SplineTopology
from .puregeometry import PureGeometry


class GoTools(PureGeometry[SplineTopology]):
    def __enter__(self) -> GoTools:
        with open(self.filename, "r") as f:
            data = f.read()
        for corners, topology, field_data in SplineTopology.from_string(data):
            self.corners.append(corners)
            self.topologies.append(topology)
            self.controlpoints.append(field_data)
        return self
