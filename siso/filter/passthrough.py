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
InS = TypeVar("InS", bound=api.Step)
OutZ = TypeVar("OutZ", bound=Zone)
OutF = TypeVar("OutF", bound=api.Field)
OutS = TypeVar("OutS", bound=api.Step)


class Passthrough(api.Source[OutF, OutS, OutZ], Generic[InF, InS, InZ, OutF, OutS, OutZ]):
    source: api.Source[InF, InS, InZ]

    def __init__(self, source: api.Source[InF, InS, InZ]):
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

    def steps(self) -> Iterator[OutS]:
        return cast(Iterator[OutS], self.source.steps())

    def zones(self) -> Iterator[OutZ]:
        return cast(Iterator[OutZ], self.source.zones())

    def topology(self, step: OutS, field: OutF, zone: OutZ) -> Topology:
        return self.source.topology(
            cast(InS, step),
            cast(InF, field),
            cast(InZ, zone),
        )

    def field_data(self, step: OutS, field: OutF, zone: OutZ) -> FieldData[floating]:
        return self.source.field_data(
            cast(InS, step),
            cast(InF, field),
            cast(InZ, zone),
        )

    def field_updates(self, step: OutS, field: OutF) -> bool:
        return self.source.field_updates(
            cast(InS, step),
            cast(InF, field),
        )
