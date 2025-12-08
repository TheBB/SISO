from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from siso import api, coord
from siso.api import (
    Points,
    Topology,
    Zone,
    ZoneShape,
    impl_basis_of,
    impl_field_data,
    impl_fields,
    impl_geometries,
    impl_topology,
)
from siso.impl import Basis, Field, Step

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from siso.util.field_data import FloatFieldData


@define
class PureGeometryZone[T: Topology]:
    corners: Points
    field_data: FloatFieldData
    topology: T
    shape: ZoneShape


class PureGeometry[T: Topology](api.Source[Basis, Field, Step, T, Zone[int]]):
    """Base class for a source that reads from a file that only has geometry.

    Subclasses should populate the `zone_data` list when `__enter__` is called.
    """

    filename: Path
    zone_data: list[PureGeometryZone[T]]

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

    @impl_basis_of
    def basis_of(self, field: Field) -> Basis:
        return Basis("mesh")

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        return
        yield

    @impl_geometries
    def geometries(self, basis: Basis) -> Iterator[Field]:
        yield Field(
            "Geometry", type=api.Geometry(self.zone_data[0].field_data.num_comps, coords=coord.Generic())
        )

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone[int]]:
        for i, zone in enumerate(self.zone_data):
            yield Zone(shape=zone.shape, coords=zone.corners, key=i)

    @impl_topology
    def topology(self, timestep: Step, basis: Basis, zone: Zone[int]) -> T:
        return self.zone_data[zone.key].topology

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FloatFieldData:
        return self.zone_data[zone.key].field_data
