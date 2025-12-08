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
from typing import TYPE_CHECKING, Self

from siso import api
from siso.api import (
    Basis,
    Field,
    Step,
    Topology,
    Zone,
    impl_basis_of,
    impl_field_data,
    impl_field_updates,
    impl_fields,
    impl_geometries,
    impl_topology,
    impl_topology_updates,
    impl_use_geometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from siso.util.field_data import FloatFieldData


# In general a filter is parametrized on ten types: the input and output B, F, S
# T and Z, respectively.
#
# For the filters that leave any of the parameters unchanged, we use just B, F,
# S, T or Z as a type variable. For those that are different, we use the 'In' or
# 'Out' variants.
#
# Thus, for example, PassthroughBFS is parametrized on B, F and S and T (which
# are unchanged), together with InZ and OutZ.
class PassthroughBase[
    InB: Basis,
    InF: Field,
    InS: Step,
    InT: Topology,
    InZ: Zone,
    OutB: Basis,
    OutF: Field,
    OutS: Step,
    OutT: Topology,
    OutZ: Zone,
](
    api.Source[OutB, OutF, OutS, OutT, OutZ],
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


class PassthroughBFST[B: Basis, F: Field, S: Step, T: Topology, InZ: Zone, OutZ: Zone](
    PassthroughBase[B, F, S, T, InZ, B, F, S, T, OutZ],
):
    """Base class for filters that change the Zone type."""

    @impl_use_geometry
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    @impl_basis_of
    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    @impl_geometries
    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    @impl_fields
    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    @impl_topology_updates
    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    @impl_field_updates
    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughBFSZ[B: Basis, F: Field, S: Step, Z: Zone, InT: Topology, OutT: Topology](
    PassthroughBase[B, F, S, InT, Z, B, F, S, OutT, Z],
):
    """Base class for filters that change the Topology type."""

    @impl_use_geometry
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    @impl_basis_of
    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    @impl_geometries
    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    @impl_fields
    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    @impl_topology_updates
    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    @impl_field_data
    def field_data(self, step: S, field: F, zone: Z) -> FloatFieldData:
        return self.source.field_data(step, field, zone)

    @impl_field_updates
    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughBFTZ[B: Basis, F: Field, T: Topology, Z: Zone, InS: Step, OutS: Step](
    PassthroughBase[B, F, InS, T, Z, B, F, OutS, T, Z],
):
    """Base class for filters that change the Step type."""

    @impl_use_geometry
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    @impl_basis_of
    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    @impl_geometries
    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    @impl_fields
    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def zones(self) -> Iterator[Z]:
        return self.source.zones()


class PassthroughBSTZ[B: Basis, S: Step, T: Topology, Z: Zone, InF: Field, OutF: Field](
    PassthroughBase[B, InF, S, T, Z, B, OutF, S, T, Z],
):
    """Base class for filters that change the Field type."""

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    @impl_topology
    def topology(self, step: S, basis: B, zone: Z) -> T:
        return self.source.topology(step, basis, zone)

    @impl_topology_updates
    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)


class PassthroughFSZ[F: Field, S: Step, Z: Zone, InB: Basis, OutB: Basis, InT: Topology, OutT: Topology](
    PassthroughBase[InB, F, S, InT, Z, OutB, F, S, OutT, Z],
):
    """Base class for filters that change the Field and Topology types."""

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    @impl_field_data
    def field_data(self, step: S, field: F, zone: Z) -> FloatFieldData:
        return self.source.field_data(step, field, zone)

    @impl_field_updates
    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class PassthroughAll[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
    PassthroughBase[B, F, S, T, Z, B, F, S, T, Z]
):
    """Base class for filters that don't change any of the type parameters."""

    @impl_use_geometry
    def use_geometry(self, geometry: F) -> None:
        self.source.use_geometry(geometry)

    def bases(self) -> Iterator[B]:
        return self.source.bases()

    @impl_basis_of
    def basis_of(self, field: F) -> B:
        return self.source.basis_of(field)

    @impl_geometries
    def geometries(self, basis: B) -> Iterator[F]:
        return self.source.geometries(basis)

    @impl_fields
    def fields(self, basis: B) -> Iterator[F]:
        return self.source.fields(basis)

    def steps(self) -> Iterator[S]:
        return self.source.steps()

    def zones(self) -> Iterator[Z]:
        return self.source.zones()

    @impl_topology
    def topology(self, step: S, basis: B, zone: Z) -> T:
        return self.source.topology(step, basis, zone)

    @impl_topology_updates
    def topology_updates(self, step: S, basis: B) -> bool:
        return self.source.topology_updates(step, basis)

    @impl_field_data
    def field_data(self, step: S, field: F, zone: Z) -> FloatFieldData:
        return self.source.field_data(step, field, zone)

    @impl_field_updates
    def field_updates(self, step: S, field: F) -> bool:
        return self.source.field_updates(step, field)


class WrappedField[F: Field](api.Field):
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
