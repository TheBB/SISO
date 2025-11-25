"""This module implements some base classes for filters. A filter is a Source
object that takes another Source as a constructor argument, and which
manipulates the data from it in some way.

All filters inherit from one of the 'Passthrough' classes in this module. The
classes are named according to which type parameters they leave unchanged. E.g.
'PassthroughBFS' is a base class intended for filters that change the Zone type
(Z), but leave the Basis (B), Field (F) and Step (S) type parameters unchanged.

In general, every passthrough class has do-nothing ("passthrough")
implementations for the methods that make sense. I.e. PassthroughBFS implements
`bases()`, which only depends on B, but not `zones()`, which depends on Z.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Self

from siso import api
from siso.api import B, F, InB, InF, InS, InT, InZ, OutB, OutF, OutS, OutT, OutZ, S, T, Z

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from numpy import floating

    from siso.util import FieldData


# In general a filter is parametrized on ten types: the input and output B, F, S
# T and Z, respectively.
#
# For the filters that leave any of the parameters unchanged, we use just B, F,
# S, T or Z as a type variable. For those that are different, we use the 'In' or
# 'Out' variants.
#
# Thus, for example, PassthroughBFS is parametrized on B, F and S and T (which
# are unchanged), together with InZ and OutZ.
class PassthroughBase(
    api.Source[OutB, OutF, OutS, OutT, OutZ],
    Generic[InB, InF, InS, InT, InZ, OutB, OutF, OutS, OutT, OutZ],
):
    """Base class for all filters. Defines the source attribute,
    together with default implementations for the Source API methods that don't
    rely on implementation details of B, F, S and Z.

    Implement validate_source() for runtime validation of source properties.
    """

    source: api.Source[InB, InF, InS, InT, InZ]

    def __init__(self, source: api.Source[InB, InF, InS, InT, InZ]):
        self.source = source
        self.validate_source()

    def validate_source(self) -> None:
        return

    def __enter__(self) -> Self:
        self.source.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.source.__exit__(exc_type, exc_val, exc_tb)

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties

    def configure(self, settings: api.ReaderSettings) -> None:
        self.source.configure(settings)

    def children(self) -> Iterator[api.Source]:
        yield self.source


class PassthroughBFST(
    PassthroughBase[B, F, S, T, InZ, B, F, S, T, OutZ],
    Generic[B, F, S, T, InZ, OutZ],
):
    """Base class for filters that change the Zone type."""

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


class PassthroughBFSZ(
    PassthroughBase[B, F, S, InT, Z, B, F, S, OutT, Z],
    Generic[B, F, S, Z, InT, OutT],
):
    """Base class for filters that change the Topology type."""

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

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        return self.source.field_data(step, field, zone)

    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughBFTZ(
    PassthroughBase[B, F, InS, T, Z, B, F, OutS, T, Z],
    Generic[B, F, T, Z, InS, OutS],
):
    """Base class for filters that change the Step type."""

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


class PassthroughBSTZ(
    PassthroughBase[B, InF, S, T, Z, B, OutF, S, T, Z],
    Generic[B, S, T, Z, InF, OutF],
):
    """Base class for filters that change the Field type."""

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    def topology(self, step: S, basis: B, zone: Z) -> T:
        return self.source.topology(step, basis, zone)

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)


class PassthroughFSZ(
    PassthroughBase[InB, F, S, InT, Z, OutB, F, S, OutT, Z],
    Generic[F, S, Z, InB, OutB, InT, OutT],
):
    """Base class for filters that change the Field and Topology types."""

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        return self.source.field_data(step, field, zone)

    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughAll(PassthroughBase[B, F, S, T, Z, B, F, S, T, Z], Generic[B, F, S, T, Z]):
    """Base class for filters that don't change any of the type parameters."""

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

    def topology(self, step: S, basis: B, zone: Z) -> T:
        return self.source.topology(step, basis, zone)

    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    def field_data(self, step: S, field: F, zone: Z) -> FieldData[floating]:
        return self.source.field_data(step, field, zone)

    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class WrappedField(api.Field, Generic[F]):
    """Base class for fields that wrap other field objects.

    Useful in many filters, so we provide a bare-bones implementation here.

    This passes all the attributes through from the `wrapped_field` attribute,
    which should be implemented by subclasses.
    """

    @property
    @abstractmethod
    def wrapped_field(self) -> F: ...

    @property
    def cellwise(self) -> bool:
        return self.wrapped_field.cellwise

    @property
    def splittable(self) -> bool:
        return self.wrapped_field.splittable

    @property
    def name(self) -> str:
        return self.wrapped_field.name

    @property
    def type(self) -> api.FieldType:
        return self.wrapped_field.type
