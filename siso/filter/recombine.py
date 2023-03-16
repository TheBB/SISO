from functools import reduce
from typing import Generic, Iterator, List, TypeVar

from attrs import define
from numpy import floating

from .. import api
from ..util import FieldData
from .passthrough import Passthrough, WrappedField


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
S = TypeVar("S", bound=api.Step)


@define
class RecombinedField(WrappedField[F]):
    sources: List[F]
    name: str

    def __post_init__(self):
        assert all(src.cellwise == self.sources[0].cellwise for src in self.sources)
        assert all(src.type == self.sources[0].type for src in self.sources)
        assert all(src.basis == self.sources[0].basis for src in self.sources)

    @property
    def original_field(self) -> F:
        return self.sources[0]

    @property
    def type(self) -> api.FieldType:
        return reduce(lambda x, y: x.concat(y), (s.type for s in self.sources))

    @property
    def splittable(self) -> bool:
        if len(self.sources) == 1:
            return self.sources[0].splittable
        return False


class Recombine(Passthrough[F, S, Z, RecombinedField[F], S, Z]):
    recombinations: List[api.RecombineFieldSpec]

    def __init__(self, source: api.Source, recombinations: List[api.RecombineFieldSpec]):
        super().__init__(source)
        self.recombinations = recombinations

    def geometries(self, basis: api.Basis) -> Iterator[RecombinedField]:
        for field in self.source.geometries(basis):
            yield RecombinedField(name=field.name, sources=[field])

    def fields(self, basis: api.Basis) -> Iterator[RecombinedField]:
        in_fields = {field.name: field for field in self.source.fields(basis)}

        for field in in_fields.values():
            yield RecombinedField(name=field.name, sources=[field])

        for spec in self.recombinations:
            if all(src in in_fields for src in spec.source_names):
                yield RecombinedField(name=spec.new_name, sources=[in_fields[src] for src in spec.source_names])

    def field_data(self, timestep: S, field: RecombinedField, zone: Z) -> FieldData[floating]:
        return FieldData.concat(self.source.field_data(timestep, src, zone) for src in field.sources)

    def field_updates(self, timestep: S, field: RecombinedField[F]) -> bool:
        return any(self.source.field_updates(timestep, src) for src in field.sources)
