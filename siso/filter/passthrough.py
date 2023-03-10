from __future__ import annotations

from typing import Generic, Iterator, TypeVar, cast

from numpy import floating
from typing_extensions import Self

from .. import api
from ..topology import Topology
from ..util import FieldData
from ..zone import Zone


InZ = TypeVar("InZ", bound=Zone)
InF = TypeVar("InF", bound=api.Field)
InT = TypeVar("InT", bound=api.TimeStep)
OutZ = TypeVar("OutZ", bound=Zone)
OutF = TypeVar("OutF", bound=api.Field)
OutT = TypeVar("OutT", bound=api.TimeStep)


class Passthrough(api.Source[OutF, OutT, OutZ], Generic[InF, InT, InZ, OutF, OutT, OutZ]):
    source: api.Source[InF, InT, InZ]

    def __init__(self, source: api.Source[InF, InT, InZ]):
        self.source = source
        self.validate_source()

    def validate_source(self) -> None:
        return

    def __enter__(self) -> Self:
        self.source.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.source.__exit__(*args)

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties

    def configure(self, settings: api.ReaderSettings) -> None:
        self.source.configure(settings)

    def use_geometry(self, geometry: OutF) -> None:
        self.source.use_geometry(cast(InF, geometry))

    def fields(self) -> Iterator[OutF]:
        return cast(Iterator[OutF], self.source.fields())

    def timesteps(self) -> Iterator[OutT]:
        return cast(Iterator[OutT], self.source.timesteps())

    def zones(self) -> Iterator[OutZ]:
        return cast(Iterator[OutZ], self.source.zones())

    def topology(self, timestep: OutT, field: OutF, zone: OutZ) -> Topology:
        return self.source.topology(
            cast(InT, timestep),
            cast(InF, field),
            cast(InZ, zone),
        )

    def field_data(self, timestep: OutT, field: OutF, zone: OutZ) -> FieldData[floating]:
        return self.source.field_data(
            cast(InT, timestep),
            cast(InF, field),
            cast(InZ, zone),
        )
