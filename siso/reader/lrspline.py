from __future__ import annotations

from ..topology import LrTopology
from .puregeometry import PureGeometry


class LrSpline(PureGeometry[LrTopology]):
    def __enter__(self) -> LrSpline:
        with open(self.filename, "r") as f:
            data = f.read()
        for corners, topology, field_data in LrTopology.from_string(data):
            self.corners.append(corners)
            self.topologies.append(topology)
            self.controlpoints.append(field_data)
        return self
