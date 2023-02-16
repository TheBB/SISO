from __future__ import annotations

from ..api import Source, SourceProperties, TimeStep, Field
from ..field import FieldData
from ..topology import Topology
from ..zone import Zone

from typing import (
    Generic,
    Iterator,
    TypeVar,
)


Z = TypeVar('Z', bound=Zone)
F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)

class Passthrough(Generic[F, T, Z]):
    source: Source[F, T, Z]

    def __init__(self, source: Source[F, T, Z]):
        self.source = source
        self.validate_source()

    def validate_source(self):
        ...

    def __enter__(self):
        self.source.__enter__()
        return self

    def __exit__(self, *args):
        self.source.__exit__(*args)

    @property
    def properties(self) -> SourceProperties:
        return self.source.properties

    def fields(self) -> Iterator[F]:
        yield from self.source.fields()

    def timesteps(self) -> Iterator[T]:
        yield from self.source.timesteps()

    def zones(self) -> Iterator[Zone]:
        yield from self.source.zones()

    def topology(self, timestep: T, field: F, zone: Z) -> Topology:
        return self.source.topology(timestep, field, zone)

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        return self.source.field_data(timestep, field, zone)
