from __future__ import annotations

from abc import abstractmethod
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

    def bases(self) -> Iterator[api.Basis]:
        return self.source.bases()

    def geometries(self, basis: api.Basis) -> Iterator[OutF]:
        return cast(Iterator[OutF], self.source.geometries(basis))

    def fields(self, basis: api.Basis) -> Iterator[OutF]:
        return cast(Iterator[OutF], self.source.fields(basis))

    def steps(self) -> Iterator[OutS]:
        return cast(Iterator[OutS], self.source.steps())

    def zones(self) -> Iterator[OutZ]:
        return cast(Iterator[OutZ], self.source.zones())

    def topology(self, step: OutS, basis: api.Basis, zone: OutZ) -> Topology:
        return self.source.topology(
            cast(InS, step),
            basis,
            cast(InZ, zone),
        )

    def topology_updates(self, step: OutS, basis: api.Basis) -> bool:
        return self.source.topology_updates(cast(InS, step), basis)

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

    def children(self) -> Iterator[api.Source]:
        yield self.source


class WrappedField(api.Field, Generic[InF]):
    @property
    @abstractmethod
    def original_field(self) -> InF:
        ...

    @property
    def cellwise(self) -> bool:
        return self.original_field.cellwise

    @property
    def splittable(self) -> bool:
        return self.original_field.splittable

    @property
    def name(self) -> str:
        return self.original_field.name

    @property
    def type(self) -> api.FieldType:
        return self.original_field.type

    @property
    def basis(self) -> api.Basis:
        return self.original_field.basis
