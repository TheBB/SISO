from __future__ import annotations

from typing import TYPE_CHECKING

from siso.api import ZoneShape
from siso.topology import LrTopology

from .puregeometry import PureGeometry, PureGeometryZone

if TYPE_CHECKING:
    from siso import api


class LrSpline(PureGeometry[LrTopology]):
    """Reader class for .lr files (LRSpline geometries)."""

    # LR-Splines don't natively support rationals. We try to deterime which
    # splines are rational based on the number of components, but this isn't
    # always possible. This setting can be set in the CLI to override this.
    rationality: api.Rationality | None = None

    def configure(self, settings: api.ReaderSettings) -> None:
        super().configure(settings)
        self.rationality = settings.rationality

    def __enter__(self) -> LrSpline:
        with self.filename.open() as f:
            data = f.read()

        # The heavy lifting is done by LrTopology.
        for corners, topology, field_data in LrTopology.from_string(data, self.rationality):
            shape = [ZoneShape.Line, ZoneShape.Quatrilateral, ZoneShape.Hexahedron][topology.pardim - 1]
            self.zone_data.append(PureGeometryZone(corners, field_data, topology, shape))
        return self
