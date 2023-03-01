from dataclasses import dataclass
from typing import Generic, Iterator, List, Optional, TypeVar

from .. import api
from ..topology import Topology
from ..util import FieldData
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


@dataclass
class DecomposedField(api.Field, Generic[F]):
    original_field: F
    components: Optional[List[int]]
    splittable: bool
    name: str

    @property
    def cellwise(self) -> bool:  # type: ignore[override]
        return self.original_field.cellwise

    @property
    def type(self) -> api.FieldType:  # type: ignore[override]
        if self.components is not None:
            if len(self.components) > 1:
                assert isinstance(self.original_field.type, api.Vector)
                return self.original_field.type.update(ncomps=len(self.components))
            return self.original_field.type.slice()
        return self.original_field.type

    @property
    def ncomps(self) -> int:
        if self.components is not None:
            return len(self.components)
        return self.original_field.ncomps


class DecomposeBase(Passthrough[F, T, Z, DecomposedField[F], T, Z]):
    def topology(self, timestep: T, field: DecomposedField, zone: Z) -> Topology:
        return self.source.topology(timestep, field.original_field, zone)

    def field_data(self, timestep: T, field: DecomposedField, zone: Z) -> FieldData:
        data = self.source.field_data(timestep, field.original_field, zone)
        if field.components is not None:
            data = data.slice(field.components)
        return data


class Decompose(DecomposeBase[F, T, Z]):
    def fields(self) -> Iterator[DecomposedField[F]]:
        for field in self.source.fields():
            yield DecomposedField(name=field.name, original_field=field, components=None, splittable=False)
            if field.is_geometry or field.is_scalar or not field.splittable:
                continue
            for i, suffix in zip(range(field.ncomps), "xyz"):
                name = f"{field.name}_{suffix}"
                yield DecomposedField(name=name, original_field=field, components=[i], splittable=False)


class Split(DecomposeBase[F, T, Z]):
    splits: List[api.SplitFieldSpec]

    def __init__(self, source: api.Source, splits: List[api.SplitFieldSpec]):
        super().__init__(source)
        self.splits = splits

    def fields(self) -> Iterator[DecomposedField[F]]:
        to_destroy = {split.source_name for split in self.splits if split.destroy}
        fields = {field.name: field for field in self.source.fields()}
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
