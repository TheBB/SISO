from __future__ import annotations

from pathlib import Path

from ..api import Source, SourceProperties
from ..field import Field, FieldType, FieldData
from ..timestep import TimeStep
from ..topology import SplineTopology
from ..zone import Zone, Shape, Coords
from .. import util

from typing import IO, Iterator, List


class GoTools(Source):
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
        with open(self.filename, 'r') as f:
            data = f.read()
        for corners, topology, data in SplineTopology.from_string(data):
            self.corners.append(corners)
            self.topologies.append(topology)
            self.controlpoints.append(data)
        return self

    def __exit__(self, *args):
        pass

    @property
    def properties(self) -> SourceProperties:
        return SourceProperties(
            instantaneous=True,
            globally_keyed=False,
        )

    def fields(self) -> Iterator[Field]:
        yield Field('Geometry', type=FieldType.Geometry, ncomps=2)

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
