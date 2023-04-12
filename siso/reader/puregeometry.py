from pathlib import Path
from typing import Generic, Iterator, List

from attrs import define
from numpy import floating

from .. import api, coord
from ..api import Points, T, Zone, ZoneShape
from ..impl import Basis, Field, Step
from ..util import FieldData


@define
class PureGeometryZone(Generic[T]):
    corners: Points
    field_data: FieldData[floating]
    topology: T


class PureGeometry(api.Source[Basis, Field, Step, T, Zone[int]], Generic[T]):
    """Base class for a source that reads from a file that only has geometry.

    Subclasses should populate the `zone_data` list when `__enter__` is called.
    """

    filename: Path
    zone_data: List[PureGeometryZone[T]]

    def __init__(self, filename: Path):
        self.filename = filename
        self.zone_data = []

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
            globally_keyed=True,
            single_basis=True,
        )

    def bases(self) -> Iterator[Basis]:
        yield Basis("mesh")

    def basis_of(self, field: Field) -> Basis:
        return Basis("mesh")

    def fields(self, basis: Basis) -> Iterator[Field]:
        return
        yield

    def geometries(self, basis: Basis) -> Iterator[Field]:
        yield Field(
            "Geometry", type=api.Geometry(self.zone_data[0].field_data.num_comps, coords=coord.Generic())
        )

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone[int]]:
        for i, zone in enumerate(self.zone_data):
            shape = [ZoneShape.Line, ZoneShape.Quatrilateral, ZoneShape.Hexahedron][zone.topology.pardim - 1]
            yield Zone(shape=shape, coords=zone.corners, key=i)

    def topology(self, timestep: Step, basis: Basis, zone: Zone[int]) -> T:
        return self.zone_data[zone.key].topology

    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FieldData[floating]:
        return self.zone_data[zone.key].field_data
