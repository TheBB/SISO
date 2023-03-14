from __future__ import annotations

from typing import Optional

from .. import api
from ..topology import LrTopology
from .puregeometry import PureGeometry


class LrSpline(PureGeometry[LrTopology]):
    rationality: Optional[api.Rationality] = None

    def configure(self, settings: api.ReaderSettings):
        super().configure(settings)
        self.rationality = settings.rationality

    def __enter__(self) -> LrSpline:
        with open(self.filename, "r") as f:
            data = f.read()
        for corners, topology, field_data in LrTopology.from_string(data, self.rationality):
            self.corners.append(corners)
            self.topologies.append(topology)
            self.controlpoints.append(field_data)
        return self
