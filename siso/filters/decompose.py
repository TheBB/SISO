from dataclasses import dataclass

from .passthrough import Passthrough
from ..field import FieldData
from ..topology import Topology
from ..api import (
    Field,
    FieldType,
    SplitFieldSpec,
    Source,
    TimeStep,
    Zone,
)

from typing import (
    ClassVar,
    Iterator,
    List,
    Optional,
    TypeVar,
)


@dataclass
class DecomposedField:
    name: str
    field: Field
    components: Optional[List[int]]
    splittable: bool

    @property
    def cellwise(self) -> bool:
        return self.field.cellwise

    @property
    def type(self) -> FieldType:
        if self.components is not None:
            return FieldType.Generic
        return self.field.type

    @property
    def ncomps(self) -> int:
        if self.components is not None:
            return len(self.components)
        return self.field.ncomps


Z = TypeVar('Z', bound=Zone)
T = TypeVar('T', bound=TimeStep)

class DecomposeBase(Passthrough[DecomposedField, T, Z]):
    def topology(self, timestep: T, field: DecomposedField, zone: Z) -> Topology:
        return self.source.topology(timestep, field.field, zone)

    def field_data(self, timestep: T, field: DecomposedField, zone: Z) -> FieldData:
        data = self.source.field_data(timestep, field.field, zone)
        if field.components is not None:
            data = data.slice(field.components)
        return data


class Decompose(DecomposeBase):
    def fields(self) -> Iterator[DecomposedField]:
        for field in self.source.fields():
            yield DecomposedField(field.name, field, components=None, splittable=False)
            if field.type == FieldType.Geometry or field.ncomps == 1 or not field.splittable:
                continue
            for i, suffix in zip(range(field.ncomps), 'xyz'):
                name = f'{field.name}_{suffix}'
                yield DecomposedField(name, field, components=[i], splittable=False)


class Split(DecomposeBase):
    splits: List[SplitFieldSpec]

    def __init__(self, source: Source, splits: List[SplitFieldSpec]):
        super().__init__(source)
        self.splits = splits

    def fields(self) -> Iterator[DecomposedField]:
        to_destroy = {split.source_name for split in self.splits if split.destroy}
        fields = {field.name: field for field in self.source.fields()}
        for field in fields.values():
            if field.name not in to_destroy:
                yield DecomposedField(field.name, field, components=None, splittable=field.splittable)
        for split in self.splits:
            yield DecomposedField(
                name=split.new_name,
                field=fields[split.source_name],
                components=split.components,
                splittable=split.splittable,
            )
