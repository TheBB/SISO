from __future__ import annotations

from siso.api import ZoneShape
from siso.topology import SplineTopology

from .puregeometry import PureGeometry, PureGeometryZone


class GoTools(PureGeometry[SplineTopology]):
    """Reader class for .g2 files (GoTools geometries)."""

    def __enter__(self) -> GoTools:
        with self.filename.open() as f:
            data = f.read()

        # The heavy lifting is done by SplineTopology.
        for corners, topology, field_data in SplineTopology.from_string(data):
            shape = [ZoneShape.Line, ZoneShape.Quatrilateral, ZoneShape.Hexahedron][topology.pardim - 1]
            self.zone_data.append(PureGeometryZone(corners, field_data, topology, shape))
        return self
