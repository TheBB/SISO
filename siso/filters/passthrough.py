from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from typing_extensions import Self

from .. import api
from ..topology import Topology
from ..util import FieldData
from ..zone import Zone


Z = TypeVar("Z", bound=Zone)
F = TypeVar("F", bound=api.Field)
T = TypeVar("T", bound=api.TimeStep)


class Passthrough(Generic[F, T, Z]):
    source: api.Source

    def __init__(self, source: api.Source):
        self.source = source
        self.validate_source()

    def validate_source(self) -> None:
        ...

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

    def fields(self) -> Iterator[F]:
        yield from self.source.fields()

    def timesteps(self) -> Iterator[T]:
        yield from self.source.timesteps()

    def zones(self) -> Iterator[Z]:
        yield from self.source.zones()

    def topology(self, timestep: T, field: F, zone: Z) -> Topology:
        return self.source.topology(timestep, field, zone)

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        return self.source.field_data(timestep, field, zone)

    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)
