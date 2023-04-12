from __future__ import annotations

from typing import Optional

from .. import api
from ..topology import LrTopology
from .puregeometry import PureGeometry, PureGeometryZone


class LrSpline(PureGeometry[LrTopology]):
    """Reader class for .lr files (LRSpline geometries)."""

    # LR-Splines don't natively support rationals. We try to deterime which
    # splines are rational based on the number of components, but this isn't
    # always possible. This setting can be set in the CLI to override this.
    rationality: Optional[api.Rationality] = None

    def configure(self, settings: api.ReaderSettings):
        super().configure(settings)
        self.rationality = settings.rationality

    def __enter__(self) -> LrSpline:
        with open(self.filename, "r") as f:
            data = f.read()

        # The heavy lifting is done by LrTopology.
        for corners, topology, field_data in LrTopology.from_string(data, self.rationality):
            self.zone_data.append(PureGeometryZone(corners, field_data, topology))
        return self
