from typing import Iterator, Set, TypeVar

from .. import api
from .passthrough import PassthroughAll


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
T = TypeVar("T", bound=api.Topology)
Z = TypeVar("Z", bound=api.Zone)


class FieldFilter(PassthroughAll[B, F, S, T, Z]):
    filters: Set[str]

    def __init__(self, source: api.Source[B, F, S, T, Z], filters: Set[str]):
        super().__init__(source)
        self.filters = filters

    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            if field.name.casefold() in self.filters:
                yield field
