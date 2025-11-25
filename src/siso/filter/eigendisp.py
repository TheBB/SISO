from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from siso import api
from siso.api import B, F, S, T, Z

from .passthrough import PassthroughBSTZ, WrappedField

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating

    from siso.util import FieldData


@define
class Wrapped(WrappedField[F]):
    """Field wrapper object that converts eigenmode fields to displacement
    fields.
    """

    wrapped_field: F

    @property
    def type(self) -> api.FieldType:
        orig_type = self.wrapped_field.type
        if not self.wrapped_field.is_eigenmode:
            return orig_type

        # Convert eigenmode fields to vector fields with displacement
        # interpretation.
        return api.Vector(
            num_comps=self.wrapped_field.num_comps,
            interpretation=api.VectorInterpretation.Displacement,
        )


class EigenDisp(PassthroughBSTZ[B, S, T, Z, F, Wrapped[F]]):
    """Filter that converts all eigenmode fields to displacement vector
    fields.
    """

    def use_geometry(self, geometry: Wrapped[F]) -> None:
        return self.source.use_geometry(geometry.wrapped_field)

    def basis_of(self, field: Wrapped[F]) -> B:
        return self.source.basis_of(field.wrapped_field)

    def geometries(self, basis: B) -> Iterator[Wrapped[F]]:
        for field in self.source.geometries(basis):
            yield Wrapped(field)

    def fields(self, basis: B) -> Iterator[Wrapped[F]]:
        for field in self.source.fields(basis):
            yield Wrapped(field)

    def field_data(self, timestep: S, field: Wrapped[F], zone: Z) -> FieldData[floating]:
        return self.source.field_data(timestep, field.wrapped_field, zone)

    def field_updates(self, timestep: S, field: Wrapped[F]) -> bool:
        return self.source.field_updates(timestep, field.wrapped_field)
