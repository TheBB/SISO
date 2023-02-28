from dataclasses import dataclass

from .passthrough import Passthrough
from ..api import (
    Field,
    FieldType,
    RecombineFieldSpec,
    Source,
    TimeStep,
    Topology,
    Zone,
)
from ..field import FieldData

from typing import Iterator, List, TypeVar


@dataclass
class RecombinedField:
    name: str
    sources: List[Field]

    def __post_init__(self):
        assert all(src.cellwise == self.sources[0].cellwise for src in self.sources)
        assert all(src.type == self.sources[0].type for src in self.sources)

    @property
    def cellwise(self) -> bool:
        return self.sources[0].cellwise

    @property
    def type(self) -> FieldType:
        return self.sources[0].type

    @property
    def ncomps(self) -> int:
        return sum(src.ncomps for src in self.sources)

    @property
    def splittable(self) -> bool:
        if len(self.sources) == 1:
            return self.sources[0].splittable
        return False


Z = TypeVar('Z', bound=Zone)
T = TypeVar('T', bound=TimeStep)

class Recombine(Passthrough[RecombinedField, T, Z]):
    recombinations: List[RecombineFieldSpec]

    def __init__(self, source: Source, recombinations: List[RecombineFieldSpec]):
        super().__init__(source)
        self.recombinations = recombinations

    def fields(self) -> Iterator[RecombinedField]:
        in_fields = {
            field.name: field
            for field in self.source.fields()
        }

        for field in in_fields.values():
            yield RecombinedField(field.name, [field])

        for spec in self.recombinations:
            yield RecombinedField(
                name=spec.new_name,
                sources=[in_fields[src] for src in spec.source_names]
            )

    def topology(self, timestep: T, field: RecombinedField, zone: Z) -> Topology:
        return self.source.topology(timestep, field.sources[0], zone)

    def field_data(self, timestep: T, field: RecombinedField, zone: Z) -> FieldData:
        return FieldData.concat(
            self.source.field_data(timestep, src, zone)
            for src in field.sources
        )
