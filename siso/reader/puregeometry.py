from pathlib import Path
from typing import Generic, Iterator, List, TypeVar

from numpy import floating

from .. import api, coord
from ..field import Field
from ..timestep import TimeStep
from ..util import FieldData
from ..zone import Coords, Shape, Zone


T = TypeVar("T", bound=api.Topology)


class PureGeometry(api.Source[Field, TimeStep, Zone], Generic[T]):
    filename: Path
    corners: List[Coords]
    controlpoints: List[FieldData[floating]]
    topologies: List[T]

    def __init__(self, filename: Path):
        self.filename = filename
        self.corners = []
        self.topologies = []
        self.controlpoints = []

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
        yield Field("Geometry", type=api.Geometry(self.controlpoints[0].ncomps, coords=coord.Generic()))

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

    def topology(self, timestep: TimeStep, field: Field, zone: Zone) -> T:
        return self.topologies[int(zone.local_key)]

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData[floating]:
        return self.controlpoints[int(zone.local_key)]
