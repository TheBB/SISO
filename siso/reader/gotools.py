from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from .. import api
from ..field import Field
from ..timestep import TimeStep
from ..topology import SplineTopology
from ..util import FieldData
from ..zone import Coords, Shape, Zone


class GoTools(api.Source[Field, TimeStep, Zone]):
    filename: Path
    corners: List[Coords]
    topologies: List[SplineTopology]
    controlpoints: List[FieldData]

    def __init__(self, filename: Path):
        self.filename = filename
        self.topologies = []
        self.controlpoints = []
        self.corners = []

    def __enter__(self) -> GoTools:
        with open(self.filename, "r") as f:
            data = f.read()
        for corners, topology, field_data in SplineTopology.from_string(data):
            self.corners.append(corners)
            self.topologies.append(topology)
            self.controlpoints.append(field_data)
        return self

    def __exit__(self, *args) -> None:
        return

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        return

    def use_geometry(self, geometry: Field) -> None:
        return

    def fields(self) -> Iterator[Field]:
        yield Field("Geometry", type=api.Geometry(self.controlpoints[0].ncomps))

    def timesteps(self) -> Iterator[TimeStep]:
        yield TimeStep(index=0)

    def zones(self) -> Iterator[Zone]:
        for i, (corners, topology) in enumerate(zip(self.corners, self.topologies)):
            shape = [Shape.Line, Shape.Quatrilateral, Shape.Hexahedron][topology.pardim - 1]
            yield Zone(
                shape=shape,
                coords=corners,
                local_key=str(i),
                global_key=None,
            )

    def topology(self, timestep: TimeStep, field: Field, zone: Zone) -> SplineTopology:
        return self.topologies[int(zone.local_key)]

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData:
        return self.controlpoints[int(zone.local_key)]
