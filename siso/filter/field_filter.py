from typing import TypeVar, Iterator, Set

from .. import api
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
S = TypeVar("S", bound=api.Step)


class FieldFilter(Passthrough[F, S, Z, F, S, Z]):
    filters: Set[str]

    def __init__(self, source: api.Source[F, S, Z], filters: Set[str]):
        super().__init__(source)
        self.filters = filters

    def fields(self) -> Iterator[F]:
        for field in self.source.fields():
            if field.name.casefold() in self.filters:
                yield field
