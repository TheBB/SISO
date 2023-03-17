from typing import Iterator, List, Optional, TypeVar

from attrs import define
from numpy import floating

from .. import api
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough, WrappedField


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
S = TypeVar("S", bound=api.Step)


@define
class DecomposedField(WrappedField[F]):
    original_field: F
    components: Optional[List[int]]
    splittable: bool
    name: str

    @property
    def type(self) -> api.FieldType:
        if self.components is not None:
            if len(self.components) > 1:
                assert isinstance(self.original_field.type, api.Vector)
                return self.original_field.type.update(ncomps=len(self.components))
            return self.original_field.type.slice()
        return self.original_field.type


class DecomposeBase(Passthrough[B, F, S, Z, B, DecomposedField[F], S, Z]):
    def use_geometry(self, geometry: DecomposedField[F]) -> None:
        return self.source.use_geometry(geometry.original_field)

    def basis_of(self, field: DecomposedField[F]) -> B:
        return self.source.basis_of(field.original_field)

    def geometries(self, basis: B) -> Iterator[DecomposedField[F]]:
        for field in self.source.geometries(basis):
            yield DecomposedField(name=field.name, original_field=field, components=None, splittable=False)

    def topology(self, timestep: S, basis: B, zone: Z) -> Topology:
        return self.source.topology(timestep, basis, zone)

    def field_data(self, timestep: S, field: DecomposedField, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(timestep, field.original_field, zone)
        if field.components is not None:
            data = data.slice(field.components)
        return data

    def field_updates(self, timestep: S, field: DecomposedField[F]) -> bool:
        return self.source.field_updates(timestep, field.original_field)


class Decompose(DecomposeBase[B, F, S, Z]):
    def fields(self, basis: B) -> Iterator[DecomposedField[F]]:
        for field in self.source.fields(basis):
            yield DecomposedField(name=field.name, original_field=field, components=None, splittable=False)
            if field.is_scalar or not field.splittable:
                continue
            for i, suffix in zip(range(field.ncomps), "xyz"):
                name = f"{field.name}_{suffix}"
                yield DecomposedField(name=name, original_field=field, components=[i], splittable=False)


class Split(DecomposeBase[B, F, S, Z]):
    splits: List[api.SplitFieldSpec]

    def __init__(self, source: api.Source, splits: List[api.SplitFieldSpec]):
        super().__init__(source)
        self.splits = splits

    def fields(self, basis: B) -> Iterator[DecomposedField[F]]:
        to_destroy = {split.source_name for split in self.splits if split.destroy}
        fields = {field.name: field for field in self.source.fields(basis)}
        for field in fields.values():
            if field.name not in to_destroy:
                yield DecomposedField(
                    name=field.name, original_field=field, components=None, splittable=field.splittable
                )
        for split in self.splits:
            yield DecomposedField(
                name=split.new_name,
                original_field=fields[split.source_name],
                components=split.components,
                splittable=split.splittable,
            )
