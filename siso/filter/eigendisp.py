from typing import Iterator, TypeVar

from attrs import define
from numpy import floating

from .. import api
from ..util import FieldData
from .passthrough import PassthroughBSZ, WrappedField


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


@define
class Wrapped(WrappedField[F]):
    original_field: F

    @property
    def type(self) -> api.FieldType:
        orig_type = self.original_field.type
        if not self.original_field.is_eigenmode:
            return orig_type
        return api.Vector(
            ncomps=self.original_field.ncomps,
            interpretation=api.VectorInterpretation.Displacement,
        )


class EigenDisp(PassthroughBSZ[B, S, Z, F, Wrapped[F]]):
    def use_geometry(self, geometry: Wrapped[F]) -> None:
        return self.source.use_geometry(geometry.original_field)

    def basis_of(self, field: Wrapped[F]) -> B:
        return self.source.basis_of(field.original_field)

    def geometries(self, basis: B) -> Iterator[Wrapped[F]]:
        for field in self.source.geometries(basis):
            yield Wrapped(field)

    def fields(self, basis: B) -> Iterator[Wrapped[F]]:
        for field in self.source.fields(basis):
            yield Wrapped(field)

    def field_data(self, timestep: S, field: Wrapped[F], zone: Z) -> FieldData[floating]:
        return self.source.field_data(timestep, field.original_field, zone)

    def field_updates(self, timestep: S, field: Wrapped[F]) -> bool:
        return self.source.field_updates(timestep, field.original_field)
