from typing import Generic, Iterator, TypeVar

from attrs import define
from numpy import floating

from .. import api
from ..util import FieldData
from .passthrough import Passthrough


Z = TypeVar("Z", bound=api.Zone)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)


@define
class Field(api.Field, Generic[F]):
    original_field: F

    @property
    def name(self) -> str:
        return self.original_field.name

    @property
    def cellwise(self) -> bool:
        return self.original_field.cellwise

    @property
    def splittable(self) -> bool:
        return self.original_field.splittable

    @property
    def type(self) -> api.FieldType:
        orig_type = self.original_field.type
        if not self.original_field.is_eigenmode:
            return orig_type
        return api.Vector(
            ncomps=self.original_field.ncomps,
            interpretation=api.VectorInterpretation.Displacement,
        )


class EigenDisp(Passthrough[F, S, Z, Field[F], S, Z]):
    def use_geometry(self, geometry: Field[F]) -> None:
        return self.source.use_geometry(geometry.original_field)

    def fields(self) -> Iterator[Field[F]]:
        for field in self.source.fields():
            yield Field(field)

    def field_data(self, timestep: S, field: Field[F], zone: Z) -> FieldData[floating]:
        return self.source.field_data(timestep, field.original_field, zone)

    def field_updates(self, timestep: S, field: Field[F]) -> bool:
        return self.source.field_updates(timestep, field.original_field)
