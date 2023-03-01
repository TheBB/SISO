from dataclasses import dataclass
from functools import reduce
from typing import Generic, Iterator, List, TypeVar

from .. import api
from ..util import FieldData
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


@dataclass
class RecombinedField(api.Field, Generic[F]):
    sources: List[F]
    name: str

    def __post_init__(self):
        assert all(src.cellwise == self.sources[0].cellwise for src in self.sources)
        assert all(src.type == self.sources[0].type for src in self.sources)

    @property
    def cellwise(self) -> bool:  # type: ignore[override]
        return self.sources[0].cellwise

    @property
    def type(self) -> api.FieldType:  # type: ignore[override]
        return reduce(lambda x, y: x.concat(y), (s.type for s in self.sources))

    @property
    def ncomps(self) -> int:
        return sum(src.ncomps for src in self.sources)

    @property
    def splittable(self) -> bool:  # type: ignore[override]
        if len(self.sources) == 1:
            return self.sources[0].splittable
        return False


class Recombine(Passthrough[F, T, Z, RecombinedField[F], T, Z]):
    recombinations: List[api.RecombineFieldSpec]

    def __init__(self, source: api.Source, recombinations: List[api.RecombineFieldSpec]):
        super().__init__(source)
        self.recombinations = recombinations

    def fields(self) -> Iterator[RecombinedField]:
        in_fields = {field.name: field for field in self.source.fields()}

        for field in in_fields.values():
            yield RecombinedField(name=field.name, sources=[field])

        for spec in self.recombinations:
            yield RecombinedField(name=spec.new_name, sources=[in_fields[src] for src in spec.source_names])

    def topology(self, timestep: T, field: RecombinedField, zone: Z) -> api.Topology:
        return self.source.topology(timestep, field.sources[0], zone)

    def field_data(self, timestep: T, field: RecombinedField, zone: Z) -> FieldData:
        return FieldData.concat(self.source.field_data(timestep, src, zone) for src in field.sources)
