from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Iterator, TypeVar

from numpy import floating
from typing_extensions import Self

from .. import api
from ..topology import Topology
from ..util import FieldData


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)
InB = TypeVar("InB", bound=api.Basis)
InF = TypeVar("InF", bound=api.Field)
InS = TypeVar("InS", bound=api.Step)
InZ = TypeVar("InZ", bound=api.Zone)
OutB = TypeVar("OutB", bound=api.Basis)
OutF = TypeVar("OutF", bound=api.Field)
OutS = TypeVar("OutS", bound=api.Step)
OutZ = TypeVar("OutZ", bound=api.Zone)


class PassthroughBase(
    api.Source[OutB, OutF, OutS, OutZ],
    Generic[InB, InF, InS, InZ, OutB, OutF, OutS, OutZ],
):
    source: api.Source[InB, InF, InS, InZ]

    def __init__(self, source: api.Source[InB, InF, InS, InZ]):
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

    def children(self) -> Iterator[api.Source]:
        yield self.source


class PassthroughSZ(
    PassthroughBase[InB, InF, S, Z, OutB, OutF, S, Z],
    Generic[S, Z, InB, OutB, InF, OutF],
):
    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()


class PassthroughBFS(
    PassthroughBase[B, F, S, InZ, B, F, S, OutZ],
    Generic[B, F, S, InZ, OutZ],
):
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughBFZ(
    PassthroughBase[B, F, InS, Z, B, F, OutS, Z],
    Generic[B, F, Z, InS, OutS],
):
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def zones(self) -> Iterator[Z]:
        return self.source.zones()


class PassthroughBSZ(
    PassthroughBase[B, InF, S, Z, B, OutF, S, Z],
    Generic[B, S, Z, InF, OutF],
):
    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    def topology(self, step: S, basis: B, zone: Z) -> Topology:
        return self.source.topology(step, basis, zone)

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)


class PassthroughAll(PassthroughBase[B, F, S, Z, B, F, S, Z], Generic[B, F, S, Z]):
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    def topology(self, step: S, basis: B, zone: Z) -> Topology:
        return self.source.topology(step, basis, zone)

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        return self.source.field_data(step, field, zone)

    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class WrappedField(api.Field, Generic[F]):
    @property
    @abstractmethod
    def original_field(self) -> F:
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
